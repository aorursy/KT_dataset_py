# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as coly

import seaborn as sns #for plotting

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import train_test_split #for data splitting

np.random.seed(123) #ensure reproducibility

import shap #for SHAP Values

from pdpbox import pdp, info_plots #for partial plots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



data_train_file = "../input/Heart_Disease_Data.csv"

data_test_file = "../input/Heart_Disease_Data.csv"



file = pd.read_csv(data_test_file)



file.head()
file.describe()
#import matplotlib.pyplot as coly

#%matplotlib inline



#For i in range(1,50):

    #sample = np.reshape(df_test [df_test.columns[1:]].iloc[i].values/255, (28,28))

    #coly.figure()

    #coly.title("label class {}" .format(df_test["label"].iloc[i]))

    #coly.imshow(sample, 'gray')
corr = file.corr()

corr.style.background_gradient()
#Calculate Z-Value - 

#Mean is 247.350 

#Stv is 51.9

#Number of disease free people is 160



y = file.loc[file['NUM'] == 1]



objects = file["SEX"].value_counts()

print(objects)



objects1 =y["SEX"].value_counts()

print(objects1)
label = 'Men','Women'

sizes = [0.6831, 0.3168]

colors = ['gold','lightskyblue']

explode = (0.1, 0) #Explore 1st Slice



#plot

coly.pie(sizes, explode = explode, labels = label, colors = colors, autopct = '%1.1f%%', shadow = True, startangle = 140)

coly.title('% of Men and Women in my dataset Cleveland Patient Heart Disease Data Analysis')

coly.axis('equal')

coly.show()
label = 'Risk of Men of Having Heart attack','Risk of Women having Heart attack'

sizes = [(0.449),(0.75)]

colors = ['gold','lightskyblue']

explode = (0.08, 0) # explode 1st slice



#plot

coly.pie(sizes, explode = explode, labels = label, colors = colors, autopct = '%1.1f%%', shadow = True, startangle = 0)

coly.axis('equal')

coly.show()
slice_hours = [207-93, 93]

activities = ['% of Men not at Risk', '% of Men at Risk']

colors = ['r','g','b','y']

coly.pie(sizes, explode = explode, labels = label, colors = colors, autopct = '%1.1f%%', shadow = True, startangle = 90)

coly.title('Analysis of Positive Heart Attach in Men Out of Total Men')

coly.show()
slice_hours = [96-72, 93]

activities = ['% of Women not at Risk', '% of Women at Risk']

colors = ['y','b']

coly.pie(sizes, explode = explode, labels = label, colors = colors, autopct = '%1.1f%%', shadow = True, startangle = 90)

coly.title('Analysis of Positive Heart Attach in Men Out of Total Women')

coly.show()
#Finding the frequency of Men and Women



n_groups = 2

means_frank = (207,96)

means_guido = (93,72)



#create plot 

fig, ax = coly.subplots()

index = np.arange(n_groups)

bar_width = 0.25

opacity = 0.6



rects1 = coly.bar(index, means_guido, bar_width, 

                alpha = opacity,

                color = 'b',

                label = 'TOTAL')



rects2 = coly.bar(index+bar_width, means_frank, bar_width, 

                alpha = opacity,

                color = 'g',

                label = 'POSITIVE WITH HEARTATTACK')



coly.xlabel('SEX')

coly.ylabel('TOTAL CASES')

coly.title('POSITIVE HEART CASES')

coly.xticks(index+bar_width, ('Men','Women'))

coly.legend()



coly.tight_layout()

coly.show()
#import the libraries 



import seaborn as sns



#Seaborn histogram



sns.distplot(file['AGE'], hist = True, kde = False, 

            bins = int (180/5), color = 'blue',

            hist_kws = {'edgecolor':'black'})

#add labels

coly.title ('Distribution of People count vs Age')

coly.xlabel('Age (in Year)')

coly.ylabel('Number of Person')

coly.show()

sns.distplot(file['AGE'])
#Seaborn histogram



sns.distplot(file['CHOL'], hist = True, kde = False, 

            bins = int (200/5), color = 'blue',

            hist_kws = {'edgecolor':'black'})

#add labels

coly.title ('Distribution of Serum Cholestoral in mg/dl')

coly.xlabel('Serum Cholestoral in mg/dl')

coly.ylabel('Count')

coly.show()

sns.distplot(file['CHOL'])
#Seaborn histogram



sns.distplot(file['THALACH'], hist = True, kde = False, 

            bins = int (180/5), color = 'blue',

            hist_kws = {'edgecolor':'black'})

#add labels

coly.title ('Distribution of Maximum Heart rate achieved')

coly.xlabel('Maximum Heart Rate Achieved')

coly.ylabel('Count')

coly.show()

sns.distplot(file['THALACH'])
#Seaborn histogram



sns.distplot(file['TRESTBPS'], hist = True, kde = False, 

            bins = int (100/5), color = 'blue',

            hist_kws = {'edgecolor':'red'})

#add labels

coly.title ('Distribution of Resting Blood Presure (in mm Hg on admission to the hospital)')

coly.xlabel('Resting Blood Pressure(in mn Hg on admission to the hospital)')

coly.ylabel('Count')

coly.show()

sns.distplot(file['TRESTBPS'])
#Seaborn histogram



sns.distplot(file['OLDPEAK'], hist = True, kde = False, 

            bins = int (180/5), color = 'blue',

            hist_kws = {'edgecolor':'black'})

#add labels

coly.title ('Distribution of ST depression induced by exercise relative to rest')

coly.xlabel('ST depression induced by exercise relative to rest')

coly.ylabel('Count')

coly.show()

## code for gaussian distbution

sns.distplot(file['OLDPEAK'])
#Seaborn histogram



sns.distplot(file['THAL'], hist = True, kde = False, 

            bins = int (180/5), color = 'blue',

            hist_kws = {'edgecolor':'black'})

#add labels

coly.title ('Distribution of St Depression Induced by exercise relative to rest')

coly.xlabel('ST depression induced by exercise relative to rest')

coly.ylabel('Count')

coly.show()

sns.distplot(file['THAL'])
# taking log chol as it gives the best gaussian distribution 



file['log_CHOL']= np.log(1+file.CHOL)
#Seaborn histogram



sns.distplot(file['log_CHOL'], hist = True, kde = False, 

            bins = int (180/5), color = 'blue',

            hist_kws = {'edgecolor':'black'})

#add labels

coly.title ('Distribution of St Depression induced by exercise relative to restPeople count vs Age')

coly.xlabel('St Depression induced by exercise relative to rest')

coly.ylabel('Count')

coly.show()

sns.distplot(file['log_CHOL'])
file ['exp_THALACH'] = np.exp(file['THALACH']/100)



#SEABORN Histogram

sns.distplot(file['exp_THALACH'], hist = True, kde = False, 

            bins = int (180/5), color = 'blue',

            hist_kws = {'edgecolor':'black'})

#add labels

coly.title ('Distribution of St Depression induced by exercise relative to rest')

coly.xlabel('St Depression induced by exercise relative to rest')

coly.ylabel('Count')

coly.show()

sns.distplot(file['exp_THALACH'])
corr = file.corr()

corr.style.background_gradient()
#### using random forest model



#Feature Scaling



#from sklearn.preprocessing import StandardScaler



#sc = StandardScaler()

#x_file = sc.fit_transform(x_file)

#x_test = sc.transform(x_file)

#from sklearn.ensemble import RandomForestRegressor



#regressor = RandomForestRegressor(n_estimators = 20, random_state=7)

#regressor.fit(x_file, x_test)

#y_pred = regressor.predict(x_test)



#for i in range(len(y_pred)):

    #if y_pred[i]>=0.5:y_pred[i]=1

   # else:

        #y_pred[i]=0

        

#from sklearn.metrics import jaccard_similarity_score

#print(jaccard_similarity_score(y_test, y_pred))



#0.92307
#X_train, X_test, Y_train, Y_test = train_test_split(file.drop('SLOPE',1),file['SLOPE'], test_si ze=.2, random_state = 10) #split the data
#model = RandomForestClassifier(max_depth=5)

#model.fixt(X_train, y_train)
#base_features = file.columns.values.tolist()

#base_features.remove('SEX')



#feat_name = 'CHOL'

#pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)



#pdp.pdp_plot(pdp_dist, feat_name)

#file.show()