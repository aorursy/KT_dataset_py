import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix,f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import validation_curve
mental_health_dataset = pd.read_csv("../input/mental-health-in-tech-survey/survey.csv")

orig_mental_health_dataset = mental_health_dataset.copy()
def understand_variables(dataset):

    print("Type = " +str(type(dataset))+"\n")

    print("Shape = "+str(dataset.shape)+"\n")

    print("Head : \n\n"+str(dataset.head())+"\n\n")

    print(str(dataset.info())+"\n\n")

    print("No.of unique values :\n\n"+str(dataset.nunique(axis=0))+"\n\n")

    print("Description :\n\n"+str(dataset.describe())+"\n\n")

    

    #print(dataset.describe(exclude=[np.number]))

    #Since no categorical variables, no need to have the above line

    

    print("Null count :\n\n"+str(dataset.isnull().sum()))

    

understand_variables(mental_health_dataset)
def understand_dist(dataset,feature_type):

    

    if feature_type == "Categorical":

        

        categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']   

        dataframes=[]

        for feature in categorical_features:

            dataframe=dataset[feature].value_counts().rename_axis(feature).reset_index(name='counts')

            dataframes.append(dataframe)

            

            plt.figure(figsize=(10,4))

            sns.countplot(x=feature,data = dataset)

            plt.show()

            print(dataframe,'\n')



        #for i in range(len(dataframes)):

        #    print(dataframes[i],'\n')

            

        

            

    elif feature_type == "Numeric":

        

        numerical_features=[feature for feature in dataset.columns if dataset[feature].dtype!='O']

        

        for feature in numerical_features:

            plt.figure(figsize=(10,4))

            sns.distplot(dataset[feature])

            plt.show()

            

            print("Description :\n\n"+str(dataset[feature].describe())+"\n\n")

understand_dist(mental_health_dataset.drop(["Timestamp"],axis=1),"Categorical")
understand_dist(mental_health_dataset,"Numeric")
mental_health_dataset = mental_health_dataset.drop(['state','comments','Timestamp','Country'],axis=1)
mental_health_dataset.loc[mental_health_dataset.self_employed.isnull(),'self_employed'] = mental_health_dataset.self_employed.mode().iloc[0]

mental_health_dataset.loc[mental_health_dataset.work_interfere.isnull(),'work_interfere'] = mental_health_dataset.work_interfere.mode().iloc[0]
print("Null count :\n\n"+str(mental_health_dataset.isnull().sum()))
#Gender



gender_dict_map = {'Male':'Male','male':'Male','male ':'Male','Female':'Female','M':'Male','female':'Female','F':'Female','m':'Male','f':'Female',

                   'Make':'Male','Woman':'Female','Male ':'Male','Female ':'Female','Man':'Male','Female (trans)':'LGBTQ','Cis Male':'Male','Male-ish':'LGBTQ','p':'LGBTQ','femail':'Female',

                   'woman':'Female','Agender':'LGBTQ','Cis Female':'Female','Guy (-ish) ^_^':'LGBTQ','Malr':'Male','Trans woman':'LGBTQ','Mail':'Male','fluid':'LGBTQ','Cis Man':'Male',

                   'Female (cis)':'Female','cis male':'Male','male leaning androgynous':'LGBTQ','queer':'LGBTQ','A little about you':'LGBTQ','Androgyne':'LGBTQ','non-binary':'LGBTQ',

                   'Trans-female':'LGBTQ','something kinda male?':'LGBTQ','Male (CIS)':'Male','queer/she/they':'LGBTQ','Genderqueer':'LGBTQ','ostensibly male, unsure what that really means':'LGBTQ',

                   'cis-female/femme':'Female','maile':'Male','All':'LGBTQ','Mal':'Male','Femake':'Female','Neuter':'LGBTQ','Nah':'LGBTQ','Enby':'LGBTQ','msle':'Male'}

mental_health_dataset.Gender = mental_health_dataset.Gender.map(gender_dict_map)
#Check if all genders covered

orig_mental_health_dataset.loc[mental_health_dataset.Gender[mental_health_dataset.Gender.isnull()].index].Gender.unique()
mental_health_dataset.head()
# Age needs to be within range of 18 to 72 



mental_health_dataset.loc[mental_health_dataset.Age<18,'Age']=18

mental_health_dataset.loc[mental_health_dataset.Age>70,'Age']=72



mental_health_true_age = mental_health_dataset.Age
# Converting categorical variables to numeric represtation



mental_health_dataset = pd.get_dummies(mental_health_dataset)
# Standard scaling (Usually from 0 to 1) is mandatory for a clustering problem



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

mental_health_dataset = pd.DataFrame(scaler.fit_transform(mental_health_dataset),columns=mental_health_dataset.columns)
# For now, we are interested in only 2 clusters



from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto',random_state=1)

kmeans = kmeans.fit(mental_health_dataset.drop(['Age'],axis=1))
cluster = []



for x in range(len(mental_health_dataset)):

    predict_me = np.array(mental_health_dataset.drop(['Age'],axis=1).iloc[x].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    cluster.append(kmeans.predict(predict_me)[0])

    
mental_health_dataset = pd.concat([mental_health_dataset,pd.Series(cluster,name='Cluster_no')],axis=1)



mental_health_dataset.Age = mental_health_true_age
cluster_1 = mental_health_dataset[mental_health_dataset.Cluster_no==0]

cluster_2 = mental_health_dataset[mental_health_dataset.Cluster_no==1]
#### Observation of Major differences b/w 2 clusters



with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    

    # We use mean to calculate % of employees who answered the index value

    cluster_compare = pd.concat([cluster_1.mean(),cluster_2.mean()],axis=1)

    cluster_compare.columns=['0','1']

    cluster_compare = cluster_compare.drop('Cluster_no')

    for index,row in cluster_compare.iterrows():

        if index!="Age":

            row['0']=row['0']*100

            row['1']=row['1']*100

    cluster_compare['Diff'] = (cluster_compare['0']-cluster_compare['1'])

    cluster_compare['Abs Diff'] = np.abs(cluster_compare['0']-cluster_compare['1'])

    cluster_compare=cluster_compare.sort_values(by='Abs Diff',ascending=False)



    

    

    print(cluster_compare)

        
#### To identify individual differences one by one b/w 2 clusters





with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    cluster_compare = pd.concat([cluster_1.mean(),cluster_2.mean()],axis=1)

    cluster_compare.columns=['0','1']

    cluster_compare = cluster_compare.drop('Cluster_no')

    for index,row in cluster_compare.iterrows():

        if index!="Age":

            row['0']=row['0']*100

            row['1']=row['1']*100

    cluster_compare['Diff'] = (cluster_compare['0']-cluster_compare['1'])

    cluster_compare['Abs Diff'] = np.abs(cluster_compare['0']-cluster_compare['1'])

    print(cluster_compare)
# PCA conversion to plot clusters succesfully in 2D



from sklearn.decomposition import PCA

reduced_data = PCA(n_components=2).fit_transform(mental_health_dataset)

results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])



plt.figure(figsize=(10,4))

sns.scatterplot(x="pca1", y="pca2", hue=mental_health_dataset['Cluster_no'], data=results)

plt.title("Employee's workspace Mental Health Attitude clustering")

plt.show()