import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



from sklearn import metrics



from itertools import combinations



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/claim_history.csv")

df.head()
from sklearn.model_selection import train_test_split

features = df[["CAR_TYPE","OCCUPATION","EDUCATION"]]

features.head()
#Split for training and testing data

features_train,features_test,labels_train, labels_test = train_test_split(features,df["CAR_USE"],test_size = 0.3, random_state=27513,stratify = df["CAR_USE"])
cross_Table_Train = pd.crosstab(labels_train,columns =  ["Count"],margins=True,dropna=True)

cross_Table_Train["Proportions"] = (cross_Table_Train["Count"]/len(labels_train))*100

cross_Table_Train
cross_Table_test = pd.crosstab(labels_test,columns =  ["Count"],margins=True,dropna=True)

cross_Table_test["Proportions"] = (cross_Table_test["Count"]/len(labels_test))*100

cross_Table_test
c=0

prob_train = len(features_train)/len(df["CAR_USE"]) #Probability of the observation in Training set

for i in df["CAR_USE"]:

    if i =="Commercial":

        c+=1        #Probability of the observation being Commercial

(prob_train*c/len(df["CAR_USE"]))/(c/len(df["CAR_USE"]))   #Probability that observation is in the Training partition given that CAR_USE = Commercial

print("The probability that an observation is in the Training partition given that CAR_USE = Commercial is",(prob_train*c/10302)/(c/10302))
count=0

prob_test = len(features_test)/len(df["CAR_USE"]) #Probability of the observation in Testing set

for i in df["CAR_USE"]:

    if i =="Private":

        count+=1        #Probability of the observation being Private

(prob_test*count/10302)/(count/10302)   #Probability that observation is in the Testing partition given that CAR_USE = Private



print("The probability that an observation is in the Testing partition given that CAR_USE = Private is",(prob_test*count/10302)/(count/10302))
features_train["Labels"] = labels_train
#Entropy of Root Node

cnt = 0

for i in df["CAR_USE"]:

    if i == "Commercial":

        cnt+=1

proba_commercial = cnt/len(df["CAR_USE"])



proba_private = (len(df["CAR_USE"])-cnt)/len(df["CAR_USE"])



ans = -((proba_commercial * np.log2(proba_commercial) + proba_private * np.log2(proba_private)))

print("Entropy for root node is given as",ans)
#All possible combinations for occupation

occupation_column = df["OCCUPATION"].unique()

occupation_combinations = []

for i in range(1,math.ceil(len(occupation_column)/2)):

    occupation_combinations+=list(combinations(occupation_column,i))
#All possible combinations for car type

car_type_column = df["CAR_TYPE"].unique()

car_type_combinations = []



for i in range(1,math.ceil(len(car_type_column)/2)+1):

    x = list(combinations(car_type_column,i))

    if i == 3:

        x = x[:10]

    car_type_combinations.extend(x) 
#All possible combinations for education

education_combinations = [("Below High School",),("Below High School","High School",),("Below High School","High School","Bachelors",),("Below High School","High School","Bachelors","Masters",)]
def EntropyIntervalSplit (

   inData,          # input data frame (predictor in column 0 and target in column 1)

   split):          # split value



   #print(split)

   dataTable = inData

   dataTable['LE_Split'] = False

   for k in dataTable.index:

       if dataTable.iloc[:,0][k] in split:

           dataTable['LE_Split'][k] = True

   #print(dataTable['LE_Split'])

   crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   

   #print(crossTable)



   nRows = crossTable.shape[0]

   nColumns = crossTable.shape[1]

   

   tableEntropy = 0

   for iRow in range(nRows-1):

      rowEntropy = 0

      for iColumn in range(nColumns):

         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]

         if (proportion > 0):

            rowEntropy -= proportion * np.log2(proportion)

      #print('Row = ', iRow, 'Entropy =', rowEntropy)

      #print(' ')

      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]

   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]

  

   return(tableEntropy)
def calculate_min_entropy(df,variable,combinations):

    inData1 = df[[variable,"Labels"]]

    entropies = []

    for i in combinations:

        EV = EntropyIntervalSplit(inData1, list(i))

        entropies.append((EV,i))

    return min(entropies)
entropy_occupation = calculate_min_entropy(features_train,"OCCUPATION",occupation_combinations)

entropy_occupation
entropy_cartype = calculate_min_entropy(features_train,"CAR_TYPE",car_type_combinations)

entropy_cartype
entropy_education = calculate_min_entropy(features_train,"EDUCATION",education_combinations)

entropy_education
df_1_left = features_train[(features_train["OCCUPATION"] == "Blue Collar") | (features_train["OCCUPATION"] == "Unknown") | (features_train["OCCUPATION"] == "Student")]

df_1_right =  features_train[(features_train["OCCUPATION"] != "Blue Collar") & (features_train["OCCUPATION"] != "Unknown") & (features_train["OCCUPATION"] != "Student")]

len(df_1_right),len(df_1_left)
left_edu_entropy = calculate_min_entropy(df_1_left,"EDUCATION",education_combinations)

left_edu_entropy
left_ct_entropy = calculate_min_entropy(df_1_left,"CAR_TYPE",car_type_combinations)

left_ct_entropy
occupation_column = ['Blue Collar', 'Unknown', 'Student']

occupation_combinations = []

for i in range(1,math.ceil(len(occupation_column)/2)):

    occupation_combinations+=list(combinations(occupation_column,i))

left_occupation_entropy = calculate_min_entropy(df_1_left,"OCCUPATION",occupation_combinations)

occupation_combinations
occupation_column = ['Professional', 'Manager', 'Clerical', 'Doctor','Lawyer','Home Maker']

occupation_combinations = []

for i in range(1,math.ceil(len(occupation_column)/2)):

    occupation_combinations+=list(combinations(occupation_column,i))

right_occupation_entropy = calculate_min_entropy(df_1_right,"OCCUPATION",occupation_combinations)



right_edu_entropy = calculate_min_entropy(df_1_right,"EDUCATION",education_combinations)

right_ct_entropy = calculate_min_entropy(df_1_right,"CAR_TYPE",car_type_combinations)

right_ct_entropy , right_edu_entropy , right_occupation_entropy
df_2_left_left = df_1_left[(features_train["EDUCATION"] == "Below High School")]

df_2_left_right = df_1_left[(features_train["EDUCATION"] != "Below High School")]
cnt = 0

for i in df_2_left_left["Labels"]:

    if i == "Commercial":

        cnt+=1

proba_commercial = cnt/len(df_2_left_left["Labels"])

print("Count of commercial and private is",cnt,(len(df_2_left_left)-cnt),"respectively and probability of the event",proba_commercial)
cnt = 0

for i in df_2_left_right["Labels"]:

    if i == "Commercial":

        cnt+=1

proba_commercial = cnt/len(df_2_left_right["Labels"])

print("Count of commercial and private is",cnt,(len(df_2_left_right)-cnt),"respectively and probability of the event",proba_commercial)
df_2_right_left = df_1_right[(features_train["CAR_TYPE"] == "Minivan") | (features_train["CAR_TYPE"] == "Sports Car") | (features_train["CAR_TYPE"] == "SUV")]

df_2_right_right = df_1_right[(features_train["CAR_TYPE"] != "Minivan") & (features_train["CAR_TYPE"] != "Sports Car") & (features_train["CAR_TYPE"] != "SUV")]
cnt = 0

for i in df_2_right_left["Labels"]:

    if i == "Commercial":

        cnt+=1

proba_commercial = cnt/len(df_2_right_left["Labels"])

1-proba_commercial

print("Count of commercial and private is",cnt,(len(df_2_right_left)-cnt),"respectively and probability of the event",proba_commercial)
cnt = 0

for i in df_2_right_right["Labels"]:

    if i == "Commercial":

        cnt+=1

proba_commercial = cnt/len(df_2_right_right["Labels"])

proba_commercial

print("Count of commercial and private is",cnt,(len(df_2_right_right)-cnt),"respectively and probability of the event",proba_commercial)
#Thresold probability of the event from training set

cnt = 0

for i in features_train["Labels"]:

    if i == "Commercial":

        cnt+=1

threshold = cnt/len(features_train["Labels"])

print("Threshold probability of an event is given as",threshold)
predicted_probability=[]

occ = ["Blue Collar","Student","Unknown"]

edu = ["Below High School",]

cartype = ["Minivan","SUV","Sports Car"]

for k in features_test.index:

    if features_test.iloc[:,1][k] in occ:

            if features_test.iloc[:,2][k] in edu:

                predicted_probability.append(0.24647887323943662)  #Leftmost Leaf Node

            else:

                predicted_probability.append(0.8504761904761905)   #Right leaf from left subtree

    else:

            if features_test.iloc[:,0][k] in cartype:

                predicted_probability.append(0.006151953245155337)  #Left leaf from right subtree

            else:

                predicted_probability.append(0.5464396284829721)   #Rightmost Leaf Node
prediction = []

for i in range(0,len(labels_test)):

    if predicted_probability[i] >= threshold :

        prediction.append("Commercial")

    else:

        prediction.append("Private")
from sklearn.metrics import accuracy_score

print("Missclassification Rate",1-accuracy_score(labels_test,prediction))
RASError = 0.0

for i in range (0,len(labels_test)):

    if labels_test.iloc[i] == "Commercial":

        RASError += (1-predicted_probability[i])**2

    else:

        RASError += (predicted_probability[i])**2

RASError = math.sqrt(RASError/len(labels_test))

RASError
true_values = 1.0 * np.isin(labels_test, ['Commercial'])

AUC = metrics.roc_auc_score(true_values, predicted_probability)

AUC
print("Root Average Squared Error",RASError)

print("Area Under the curve",AUC)

print("Missclassification Rate",1-accuracy_score(labels_test,prediction))
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(labels_test, predicted_probability, pos_label = 'Commercial')
OneMinusSpecificity = np.append([0], OneMinusSpecificity)

Sensitivity = np.append([0], Sensitivity)



OneMinusSpecificity = np.append(OneMinusSpecificity, [1])

Sensitivity = np.append(Sensitivity, [1])
plt.figure(figsize=(6,6))

plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',

         color = 'orange', linestyle = 'solid', linewidth = 2, markersize = 6)

plt.plot([0, 1], [0, 1], color = 'darkblue', linestyle = '--')

plt.grid(True)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Receiver Operating Characteristic Curve")

ax = plt.gca()

ax.set_aspect('equal')

plt.show()