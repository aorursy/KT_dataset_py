# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings('ignore')
def plot_pricision_recall(model,X_test,y_test,title_name):

    # ------ get probability of Prediction for true result only using predict_proba func. ------

    y_pred = model.predict_proba(X_test)[:,1]



    # ------ get precision,recall and thresholds to Drawing the curve to clarify result --------

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    

    # --------------------- Intialize Graph ---------------------

    plt.xlabel("Threshhold")

    plt.ylabel("Precision/Recall")

    plt.title('Precision/Recall Graph '+title_name)



    # --------------------- Drawing the graph ---------------------

    plt.plot(thresholds,precision[:-1],label='Precision')

    plt.plot( thresholds,recall[:-1], label='Recall')



    # --------------------- Set Legend on the graph ---------------------

    plt.legend(bbox_to_anchor=(1.28,1), loc='best', borderaxespad=0.)

# --------------------- Load Dataset using panda ---------------------

data = pd.read_csv("/kaggle/input/autism-screening/Autism_Data.arff") 



# --------------------- Remove missing values in panda using dropna func. ---------------------

data = data.dropna()



# --------------------- Adjust columns names for dataset ---------------------

data.columns = [

    "A1_Score",

    "A2_Score",

    "A3_Score",

    "A4_Score",

    "A5_Score",

    "A6_Score",

    "A7_Score",

    "A8_Score",

    "A9_Score",

    "A10_Score",

    "age",

    "gender",

    "ethnicity",

    "jaundice",

    "autism",

    "country_of_res",

    "used_app_before",

    "result",

    "age_desc",

    "relation",

    "Class_ASD",

]



# --------------------- Print first 10 examples in dataset to explore it ---------------------

data.head()
# --------------------- Filtering dataset fron case number column ---------------------

data = data[data['age']!='?']
# --------------------- Print out the unique values for each column ---------------------

for column_name in data.columns:

    print(

        """{column_name}:

        {unique_values}""".format(

            column_name=column_name,

            unique_values=", ".join(

                map(str, data[column_name].unique())

            ),

        )

    )

    

print(

    """

NUMBER OF EXAMPLES:{}

NUMBER OF COLUMNS: {}

""".format(

        data.shape[0], data.shape[1]

    )

)
# - Draw histogram for each column with class/asd to see which columns Influential in class/ASD -

bin_data=pd.DataFrame.copy(data)



# --------------------- Encode Class/ASD_Traits unique values to 1 or 0 ---------------------

bin_data['Class_ASD']=bin_data['Class_ASD'].apply(lambda x:1 if x=="YES" else 0)



for column_name in data.columns:

    if(column_name == 'Class_ASD'):

        continue;

    group_by_modelLine = bin_data[[column_name,'Class_ASD']].groupby(by=[column_name])

# --------------------- get mean to draw histogram ---------------------    

    mean=group_by_modelLine.mean().reset_index()

    mean.columns=[column_name,'positive']

    mean['negative']=mean['positive'].apply(lambda x: 1-x)

    mean=pd.melt(mean, id_vars=column_name, var_name="Pos/Neg", value_name="Class_ASD")

    p=sns.barplot(x=column_name, y='Class_ASD', hue='Pos/Neg', data=mean.reset_index())

    p.legend(loc='best', bbox_to_anchor=(1.28, 1), ncol=1)

    plt.show()
# --------------------- The columns will not change ---------------------

org_data = data[

    [

        "A1_Score",

        "A2_Score",

        "A3_Score",

        "A4_Score",

        "A5_Score",

        "A6_Score",

        "A7_Score",

        "A8_Score",

        "A9_Score",

        "A10_Score",

        "age",

    ]

]



# --------------------- The columns wich will be encoded using label encoding -------------------

label_encoded_data = data[

    ["gender", "autism", "jaundice", "Class_ASD"]

]

label_encoded_data["gender"] = label_encoded_data[

    "gender"

].apply(lambda x: 1 if x == "m" else 0)

label_encoded_data["autism"] = label_encoded_data[

    "autism"

].apply(lambda x: 1 if x == "yes" else 0)

label_encoded_data["jaundice"] = label_encoded_data[

    "jaundice"

].apply(lambda x: 1 if x == "yes" else 0)

label_encoded_data["Class_ASD"] = label_encoded_data[

    "Class_ASD"

].apply(lambda x: 1 if x == "YES" else 0)



# --------------------- The columns wich will be encoded using one hot encoding -----------------

one_hot_encoded_data = data[["ethnicity"]]

one_hot_encoded_data = pd.get_dummies(one_hot_encoded_data)
# --------------------- concatenate all columns ---------------------

fixed_data = pd.concat([org_data,label_encoded_data,one_hot_encoded_data],axis=1)



# --------------------- Droping column Class/ASD_Traits from features ---------------------

X = fixed_data.drop(columns=['Class_ASD'])

y = fixed_data[['Class_ASD']]
# --------------------- Splitting data for test data and train data ---------------------

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



# --------------------- Create LogisticRegression model and train it ---------------------

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)

plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(clf,X_train,y_train,'LogisticRegression-Train')

plt.subplot(122)

plot_pricision_recall(clf,X_val,y_val,'LogisticRegression-Validation')

plt.show()
# --------------------- RandomForestClassifier ---------------------

rfc= RandomForestClassifier(n_estimators=500)

rfc.fit(X_train,y_train)



plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(rfc,X_train,y_train,'RandomForest-Train')

plt.subplot(122)

plot_pricision_recall(rfc,X_val,y_val,'RandomForest-Validation')

plt.show()
knn= KNeighborsClassifier(n_neighbors=13)

knn.fit(X_train,y_train)



plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(knn,X_train,y_train,'KNearestNeighbours-Train')

plt.subplot(122)

plot_pricision_recall(knn,X_val,y_val,'KNearestNeighbours-Validation')

plt.show
plt.subplots(1,3,figsize=(20,5))

plt.subplots_adjust(wspace=0.4)

plt.subplot(131)

plot_pricision_recall(clf,X_test,y_test,'LogisticRegression-Test')

plt.subplot(132)

plot_pricision_recall(rfc,X_test,y_test,'RandomForest-Test')

plt.subplot(133)

plot_pricision_recall(knn,X_test,y_test,'KNearestNeighbours-Test')
