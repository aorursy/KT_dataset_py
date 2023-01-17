import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
import io
import seaborn as sns

#cervcads = pd.read_csv('gdrive/My Drive/ccancer.csv')
cervcads=pd.read_csv('../input/cervicalcancer.csv')
cervcads.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'],axis=1,inplace=True)
numerical_ds = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)']
categorical_ds = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',
                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS', 
                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN', 
                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']
cervcads = cervcads.replace('?', np.NaN)
for feature in numerical_ds:
    print(feature,'',pd.to_numeric(cervcads[feature]).median())
    feature_median = round(pd.to_numeric(cervcads[feature]).median())
    cervcads[feature]= cervcads[feature].fillna(feature_median)

for feature in categorical_ds:
    cervcads[feature] = pd.to_numeric(cervcads[feature]).fillna(1.0)
cervcads['Number of sexual partners'] = round(pd.to_numeric (cervcads['Number of sexual partners']))
cervcads['First sexual intercourse'] = pd.to_numeric(cervcads['First sexual intercourse'])
cervcads['Num of pregnancies']=round(pd.to_numeric(cervcads['Num of pregnancies']))
cervcads['Smokes'] = pd.to_numeric(cervcads['Smokes'])
cervcads['Smokes (years)'] = pd.to_numeric(cervcads['Smokes (years)'])
cervcads['Hormonal Contraceptives'] = pd.to_numeric(cervcads['Hormonal Contraceptives'])
cervcads['Hormonal Contraceptives (years)'] = pd.to_numeric(cervcads['Hormonal Contraceptives (years)'])
cervcads['IUD (years)'] = pd.to_numeric(cervcads['IUD (years)'])
cervcads['Smokes (packs/year)'] = pd.to_numeric(cervcads['Smokes (packs/year)'])
cervcads['STDs (number)'] = pd.to_numeric(cervcads['STDs (number)'])
cervcads.drop('Hormonal Contraceptives',axis=1,inplace=True)
cervcads.drop('Dx',axis=1,inplace=True)
cervcads=cervcads.drop(['IUD', 'STDs', 'STDs:condylomatosis','Smokes (packs/year)',
       'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
       'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
       'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
       'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
       'STDs: Number of diagnosis', 'Dx:CIN', 'STDs:HPV','STDs:Hepatitis B'],axis=1)
cervcads.shape
# Shuffle the Dataset.
shuffled_df = cervcads.sample(frac=1,random_state=4)

# Put all the biopsy class in a separate dataset.
positive_df = shuffled_df.loc[shuffled_df['Biopsy'] == 1]

#Randomly select 400 observations from the majority class
negative_df = shuffled_df.loc[shuffled_df['Biopsy'] == 0].sample(n=400,random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([positive_df, negative_df])
cervcads=normalized_df
#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))
sns.countplot('Biopsy', data=normalized_df)
plt.title('Balanced Classes')
plt.show()
cervcads.shape
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_val_score, train_test_split

X = np.array(cervcads.drop('Biopsy',1))
#X = preprocessing.scale(X)
y = np.array(cervcads['Biopsy'])

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X=scaler.transform(X)
cervcads.info()
accuracy = []
x_range = []
for j in range(1000):
    x_range.append(j)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    acc = knn.score(X_test,y_test)
    accuracy.append(acc)
    
plt.title(str(5) + ' nearest neighbors')
plt.plot(x_range, accuracy)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
test_label = np.array(cervcads['Biopsy'])
    
predictions = knn.predict(X_test)
#predictions = knn.predict(X_test)
print(predictions)
print(y_test)
print('With KNN (K=5) accuracy is: ',knn.score(X_test,y_test)) # accuracy
df_ans = pd.DataFrame({'Biopsy' :y_test})
df_ans['predictions'] = predictions
from sklearn.metrics import classification_report
knn_pred = knn.predict(X_test)
print(classification_report(y_test, knn_pred))
plt.show()#Evaluating the classifier using training set
from sklearn.metrics import accuracy_score
y_pred=knn.predict(X_test)
accuracy_score(y_pred, y_test)
df_ans['Biopsy'].value_counts()
df_ans['predictions'].value_counts()
cols = ['Biopsy_1','Biopsy_0']  #Gold standard
rows = ['Prediction_1','Prediction_0'] #diagnostic tool (our prediction)

B1P1 = len(df_ans[(df_ans['predictions'] == df_ans['Biopsy']) & (df_ans['Biopsy'] == 1)])
B1P0 = len(df_ans[(df_ans['predictions'] != df_ans['Biopsy']) & (df_ans['Biopsy'] == 1)])
B0P1 = len(df_ans[(df_ans['predictions'] != df_ans['Biopsy']) & (df_ans['Biopsy'] == 0)])
B0P0 = len(df_ans[(df_ans['predictions'] == df_ans['Biopsy']) & (df_ans['Biopsy'] == 0)])

conf = np.array([[B1P1,B0P1],[B1P0,B0P0]])
df_cm = pd.DataFrame(conf, columns = [i for i in cols], index = [i for i in rows])

f, ax= plt.subplots(figsize = (5, 5))
sns.heatmap(df_cm, annot=True, ax=ax) 
ax.xaxis.set_ticks_position('top') #Making x label be on top is common in textbooks.

print('total test case number: ', np.sum(conf))


def model_efficacy(conf):
    
    total_num = np.sum(conf)
    sen = conf[0][0]/(conf[0][0]+conf[1][0])
    spe = conf[1][1]/(conf[1][0]+conf[1][1])
    false_positive_rate = conf[0][1]/(conf[0][1]+conf[1][1])
    false_negative_rate = conf[1][0]/(conf[0][0]+conf[1][0])
    
    print('total_num: ',total_num)
    print('G1P1: ',conf[0][0]) 
    print('G0P1: ',conf[0][1])
    print('G1P0: ',conf[1][0])
    print('G0P0: ',conf[1][1])
    print('##########################')
    print('sensitivity: ',sen)
    print('specificity: ',spe)
    print('false_positive_rate: ',false_positive_rate)
    print('false_negative_rate: ',false_negative_rate)
    
    return total_num, sen, spe, false_positive_rate, false_negative_rate

model_efficacy(conf)
#testing  raw data predictions
Age = 20
Number_of_sexual_partners = 2
First_sexual_intercourse = 15
Num_of_pregnancies = 2
SmokesYears = 0
SmokesPacks = 0
HormonalContraceptives = 0.5
IUD =0
STDs =0
Cancer =0
HPV =0
Hinselmann =0
Citology =0
Schiller=0


#now create an np array for the above info provided by the user through an interface app
#start with a python list
user_input_list = [
  Age,
  Number_of_sexual_partners,
  First_sexual_intercourse,
  Num_of_pregnancies,
  SmokesYears,
  SmokesPacks,
  HormonalContraceptives,
  IUD ,
  STDs,
  Cancer,
  HPV ,
  Hinselmann,
  Citology,
  Schiller,
      
]
#then convert list to numpy array
user_input = np.array(user_input_list)

print('np array')
print(user_input)


#reshape to 2d array --> to suit data dimension
user_input = np.reshape(user_input, (-1, 14))
print('2D np array')
print(user_input)



print()


user_input.shape
predictions = knn.predict(user_input)
print(predictions)
result = True
if  predictions == 0:
    result = False
print('prediction of developing cancer: ' + str(result))
# Model persistence
#output_model_file = 'finalknn1model.pkl'
# Save the model
#with open(output_model_file, 'wb') as f:
 #  pickle.dump(knn, f)
#My first machine learning project after viewing what others have done and researching a bit.
#First attempt of the prediction model for my Cervapp project used undersampling and stratify as dataset is imbalanced and skewed next try will be using SMOTE
#compare the 2.