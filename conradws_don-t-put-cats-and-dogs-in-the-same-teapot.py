

import pandas as pd #Pandas is the most popular library for manipulating data. Think of it as an Excel but a million times faster and more practical.

import numpy as np # This library allows to easily carry out simple and complex mathematical operations.

import matplotlib.pyplot as plt #Allows us to plot data, create graphs and visualize data. Perfect for your Powerpoint slides ;)

import sklearn #The one and only. This amazing library holds all the secrets. Containing powerful algorithms packed in a single line of code, this is where the magic will happen.

import sklearn.model_selection # more of sklearn. It is a big library, but trust me it is worth it.

import sklearn.preprocessing 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, explained_variance_score,mean_absolute_error,mean_squared_error,precision_score,recall_score, accuracy_score,f1_score

from sklearn.utils import shuffle

import os





import random # Allows us to call random numbers, occasionally very useful.

import pprint#Allows us to neatly display text

from collections import OrderedDict



import seaborn as sns

import numpy as np

import pandas as pd

from sklearn.metrics.scorer import make_scorer

from sklearn.metrics import confusion_matrix



def quadratic_kappa(y_true, y_pred):

    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition

    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 

    of adoption rating."""

    w = np.zeros((5,5))

    O = confusion_matrix(y_true, y_pred)

    for i in range(len(w)): 

        for j in range(len(w)):

            w[i][j] = float(((i-j)**2)/(5-1)**2)

    

    act_hist=np.zeros([5])

    for item in y_true: 

        act_hist[item]+=1

    

    pred_hist=np.zeros([5])

    for item in y_pred: 

        pred_hist[item]+=1

                         

    E = np.outer(act_hist, pred_hist);

    E = E/E.sum();

    O = O/O.sum();

    

    num=0

    den=0

    for i in range(len(w)):

        for j in range(len(w)):

            num+=w[i][j]*O[i][j]

            den+=w[i][j]*E[i][j]

    return (1 - (num/den))
def plot_correlation_matrix(df):

    corr = df.corr()



    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(20, 9))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})
#Let's load our data

print(os.listdir("../input"))



data = pd.read_csv("../input/train/train.csv")

data.columns
data.set_index('PetID', inplace=True)

#I'm not going to use NLP in this Kernel, soz!

data.drop(columns=['RescuerID','Name','Description'], inplace=True)



#Shuffling data... Just in case

data= data.sample(frac=1, random_state=85)



data.head() 
states = pd.read_csv('../input/state_labels.csv')





states.at[0, "State Urbanisation"] = 71.9

states.at[1, "State Urbanisation"] = 64.6

states.at[2, "State Urbanisation"] = 42.4

states.at[3, "State Urbanisation"] = 100

states.at[4, "State Urbanisation"] = 82.3

states.at[5, "State Urbanisation"] = 86.5

states.at[6, "State Urbanisation"] = 66.5

states.at[7, "State Urbanisation"] = 50.5

states.at[8, "State Urbanisation"] = 69.7

states.at[9, "State Urbanisation"] = 51.4

states.at[10, "State Urbanisation"] = 90.8

states.at[11, "State Urbanisation"] = 54

states.at[12, "State Urbanisation"] = 53.8

states.at[13, "State Urbanisation"] = 91.4

states.at[14, "State Urbanisation"] = 59.1



states.at[0, "State Population"] = 3348283

states.at[1, "State Population"] = 1890098

states.at[2, "State Population"] = 1459994

states.at[3, "State Population"] = 1627172

states.at[4, "State Population"] = 86908

states.at[5, "State Population"] = 788706

states.at[6, "State Population"] = 997071

states.at[7, "State Population"] = 1443365

states.at[8, "State Population"] = 2258428

states.at[9, "State Population"] = 227025

states.at[10, "State Population"] = 1520143

states.at[11, "State Population"] = 3117405

states.at[12, "State Population"] = 72420009

states.at[13, "State Population"] = 5411324

states.at[14, "State Population"] = 1015776.9









states
#We convert the label dataframe into a dictionary that we then map onto our train dataframe:

states_ubran_dict=states.set_index('StateID')['State Urbanisation'].to_dict()

states_pop_dict=states.set_index('StateID')['State Population'].to_dict()



data["State Urbanisation"]=data["State"].map(states_ubran_dict)

data["State Population"]=data["State"].map(states_pop_dict)

data.drop(columns="State", inplace= True)





data.head()

breeds=states = pd.read_csv('../input/breed_labels.csv')



breeds.at[0,"Ranked Popularity"]= 147

#Let's speed this up. Here is a list of the ranked popularity with the same index as the breed labels.

breeds_popularity_list=[147,93,55,190,47,59,5,118,126,190,148,86,190,56,190,17,137,84,39,6,125,141,151,120,190,44,190,25,46,130,1,190,116,50,190,132,124,190,38,92,102,21,85,11,98,127,26,94,60,51,69,175,37,190,190,190,24,19,43,32,79,190,179,1,76,184,140,190,29,40,130,15,81,171,13,63,182,16,190,67,190,5,52,178,113,95,70,27,134,152,118,190,145,190,173,177,96,99,187,4,156,136,2,10,190,62,80,158,3,104,14,66,75,156,183,23,160,190,12,165,189,72,114,164,73,74,111,78,119,190,190,190,190,87,129,190,190,163,157,190,1,138,190,100,77,169,33,70,190,28,190,71,190,152,190,107,190,36,128,170,91,190,105,89,70,190,53,190,88,190,160,174,190,166,159,113,176,22,7,54,190,31,142,162,97,139,96,41,8,48,123,57,190,110,90,168,58,150,104,64,140,70,24,45,20,12,106,172,188,122,190,90,177,60,7,161,149,107,190,153,121,101,190,112,131,30,34,15,133,109,42,49,61,2,99,65,70,143,1,9,200,10,22,6,42,12,28,18,16,34,31,3,20,39,5,50,36,41,50,13,50,9,5,5,50,50,7,25,1,50,38,42,24,28,37,42,6,29,50,50,14,21,11,11,11,4,6,26,2,17,7,23,12,15,50,30,50,27,8,50,50,19,50,10,33,41,45]

i=0

#With a simple "for loop", we can add the whole list the breed_label dataframe

for i in range (len(breeds_popularity_list)):

        breeds.at[i,"Ranked Popularity"]= breeds_popularity_list[i]

breeds
breeds_dictionary=breeds.set_index('BreedID')['Ranked Popularity'].to_dict()



#We map the dictionary onto the train dataframe

data["Breed1"]=data["Breed1"].map(breeds_dictionary)

data["Breed2"]=data["Breed2"].map(breeds_dictionary)



#if we  only have one breed for both columns, then we'll repeat the bread, indicating that this is a pure breed

data["Breed2"].fillna(data["Breed1"],inplace=True)

data["Breed1"].fillna(data["Breed2"],inplace=True)





data.head()
colors = pd.read_csv('../input/color_labels.csv')





colors.at[0, "Color Intensity"] = 1

colors.at[1, "Color Intensity"] = 0.883

colors.at[2, "Color Intensity"] = 0.667

colors.at[3, "Color Intensity"] = 0.333

colors.at[4, "Color Intensity"] = 0.167

colors.at[5, "Color Intensity"] = 0.5

colors.at[6, "Color Intensity"] = 0



colors
color_dictionary=colors.set_index('ColorID')['Color Intensity'].to_dict()





data["Color2"]= data["Color2"].apply(lambda x: None if x==0 else x)

data["Color3"]= data["Color3"].apply(lambda x: None if x==0 else x)





data["Color1"]=data["Color1"].map(color_dictionary)

data["Color2"]=data["Color2"].map(color_dictionary)

data["Color3"]=data["Color3"].map(color_dictionary)



#This always to standerize the colors, if color1=color2=color3, then the pet only has one solid color

data["Color2"].fillna(data["Color1"],inplace=True)

data["Color3"].fillna(data["Color2"],inplace=True)



#for the second row (id=29ffe21fe), we can see that that the pet is black and white

data.head()
data["Vaccinated"]=data["Vaccinated"].apply(lambda x:1 if x==1 else 0)

data["Dewormed"]=data["Dewormed"].apply(lambda x:1 if x==1 else 0)

data["Sterilized"]=data["Sterilized"].apply(lambda x:1 if x==1 else 0)

#data["Gender"]=data["Gender"].apply(lambda x:1 if x==2 else 0)





data.head()
#Let's just save our formatted data just in case.



data.to_csv("formated_data_petfinder.csv")

#creating our x datasheet, which contains all our data except for the ['AdoptionSpeed'] column.

x=data.drop(columns='AdoptionSpeed')





#The ['AdoptionSpeed'] column extracted.

y=data['AdoptionSpeed']
#Checking the features that are correlated to the target feature (Adoption speed)

for columnname in x.columns: 

    if abs(x[columnname].corr(y)) >0.095 :    

      print('The Correlation Between',columnname,'and Adoption Speed is significant:', abs(x[columnname].corr(y)))
from sklearn.ensemble import ExtraTreesClassifier

feature_importance_indicator=ExtraTreesClassifier(n_estimators = 100)



#Then we ask this algorithm, now dubbed as "feature_importance_indicator", to learn from our data x ( the indicators) and y( the Adoption Speed outcome):



feature_importance_indicator.fit(x,y)



#We then ask the model politely to create a list of which columns it learnt the most from and which columns didn't help it at all, for this we use the "feature_importances_" command:



importance_dict= dict(zip(x.columns, (feature_importance_indicator.feature_importances_)))



#We sort the list in descending order so that it is quicker to analyse.



importance_dict_sorted= OrderedDict(sorted(importance_dict.items(), key=lambda x:x[1],reverse=1))



#The "pprint" commnad allows us to print things out nicely.



pprint.pprint(importance_dict_sorted)
features = x.columns

importances = feature_importance_indicator.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='r', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Importance Level')

plt.show()
plot_correlation_matrix(data[data.Type==1].drop(columns=["Type"]))

plot_correlation_matrix(data[data.Type==2].drop(columns=["Type"]))

rad_viz = pd.plotting.radviz(x, 'Type')


#we create two datasets, one for dogs and one for cats to train seperate data on them.



dogdata = data[data.Type==1]

catdata = data[data.Type==2]





dog_x=dogdata.drop(columns=["AdoptionSpeed"])

dog_y=dogdata["AdoptionSpeed"]



cat_x=catdata.drop(columns=["AdoptionSpeed"])

cat_y=catdata["AdoptionSpeed"]





dog_x.columns
feature_importance_indicator.fit(dog_x,dog_y)

features = dog_x.columns

importances = feature_importance_indicator.feature_importances_

indices = np.argsort(importances)



plt.title('Dog Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='r', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Importance Level')

plt.show()
feature_importance_indicator.fit(cat_x,cat_y)

features = cat_x.columns

importances = feature_importance_indicator.feature_importances_

indices = np.argsort(importances)



plt.title('Cat Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='r', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Importance Level')

plt.show()


#splitting between train and test for the dog data

dog_x_training,dog_x_testing, dog_y_training, dog_y_testing = train_test_split(dog_x, dog_y, test_size=0.1, random_state=85)



#splitting between train and test for the cat data

cat_x_training,cat_x_testing,  cat_y_training, cat_y_testing = train_test_split(cat_x, cat_y, test_size=0.1, random_state=85)



dog_x_training.drop(columns="Type", inplace=True)

dog_x_testing.drop(columns="Type",inplace=True)

cat_x_training.drop(columns="Type",inplace=True)

cat_x_testing.drop(columns="Type",inplace=True)

import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline





# Average CV score on the training set was:0.368919785590269



dog_best_model = make_pipeline(

    VarianceThreshold(threshold=0.0001),

    GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=0.5, min_samples_leaf=4, min_samples_split=17, n_estimators=100, subsample=1.0)

)





dog_best_model.fit(dog_x_training, dog_y_training)



y_true_train= np.array(dog_y_training)

y_pred_train = np.array(dog_best_model.predict(dog_x_training))



y_true_test= np.array(dog_y_testing)

y_pred_test = np.array(dog_best_model.predict(dog_x_testing))





print("On training data QWK:...",quadratic_kappa(y_true_train,y_pred_train),"on testing data QWK...",quadratic_kappa(y_true_test,y_pred_test))

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier





# Average CV score on the training set was:0.31768177281003007

cat_best_model = make_pipeline(

    StandardScaler(),

    XGBClassifier(learning_rate=0.1, max_depth=4, min_child_weight=13, n_estimators=100, nthread=1, subsample=0.7500000000000001)

)



cat_best_model.fit(cat_x_training,cat_y_training)

y_true_train= np.array(cat_y_training)

y_pred_train = np.array(cat_best_model.predict(cat_x_training))



y_true_test= np.array(cat_y_testing)

y_pred_test = np.array(cat_best_model.predict(cat_x_testing))



print("On training data...",quadratic_kappa(y_true_train,y_pred_train),"on testing data...",quadratic_kappa(y_true_test,y_pred_test))

#Let's first retrain our models on the complete dataset

dog_best_model.fit(dog_x, dog_y)

cat_best_model.fit(cat_x,cat_y)

#Loading the test data provided by Kaggle

test_data=pd.read_csv("../input/test/test.csv")

#Formatting the test data just like we did with the train data...

test_data.drop(columns=['RescuerID','Name','Description','PetID'], inplace=True)



test_data["State urbanisation"]=test_data["State"].map(states_ubran_dict)

test_data["State population"]=test_data["State"].map(states_pop_dict)

test_data.drop(columns="State", inplace= True)



test_data["Color2"]= test_data["Color2"].apply(lambda x: None if x==0 else x)

test_data["Color3"]= test_data["Color3"].apply(lambda x: None if x==0 else x)





test_data["Color1"]=test_data["Color1"].map(color_dictionary)

test_data["Color2"]=test_data["Color2"].map(color_dictionary)

test_data["Color3"]=test_data["Color3"].map(color_dictionary)



test_data["Color2"].fillna(test_data["Color1"],inplace=True)

test_data["Color3"].fillna(test_data["Color2"],inplace=True)



test_data["Breed1"]=test_data["Breed1"].map(breeds_dictionary)

test_data["Breed2"]=test_data["Breed2"].map(breeds_dictionary)



test_data["Breed2"].fillna(test_data["Breed1"],inplace=True)

test_data["Breed1"].fillna(test_data["Breed2"],inplace=True)



test_data["Vaccinated"]=test_data["Vaccinated"].apply(lambda x:1 if x==1 else 0)

test_data["Dewormed"]=test_data["Dewormed"].apply(lambda x:1 if x==1 else 0)

test_data["Sterilized"]=test_data["Sterilized"].apply(lambda x:1 if x==1 else 0)









test_data.head()
#Making sure the formatting was done correctly

if len(test_data.columns)==len(dog_x.columns):

  print("Formatting done correctly")

else:

    print("H we have a problem, double check previous cell")
#We'll store our answers in this dictionary so that we can map them onto the submission file

answers_dictionary={}



def test_predicitions(test_data):

    assert type(test_data)==type(data)

    for i in range (len(test_data)):

        if int(test_data["Type"].iloc[[i,]])==1:

            test_data.iloc[[i,]].drop(columns="Type")

            answers_dictionary[test_data.index[i]]=int(dog_best_model.predict(test_data.iloc[[i,]]))

        else:

            test_data.iloc[[i,]].drop(columns="Type")

            answers_dictionary[test_data.index[i]]=int(cat_best_model.predict(test_data.iloc[[i,]]))



test_predicitions(test_data)     

submission_answer=pd.read_csv("../input/test/sample_submission.csv")



submission_answer.head()



#we map our asnwers from our two different models onto the submission file

submission_answer["AdoptionSpeed"]= submission_answer.index.map(answers_dictionary.get)



submission_answer.head()
submission_answer.to_csv('submission.csv', index=False)