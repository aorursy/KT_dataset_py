import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
def fl(x):

    try:

        return float(x)

    except ValueError:

        return 0
def preprocess1(df):

    df['TotalCharges'] = df['TotalCharges'].apply(lambda x: fl(x))

    channels = ['Channel1','Channel2','Channel3','Channel4','Channel5','Channel6']

    df.loc[df['Channel1']=='No tv connection',channels] = 'No'

    df.loc[df['HighSpeed']=='No internet','HighSpeed'] = 'No'

    for i in range(0,df.shape[0]):

        if(df.loc[i,'TVConnection'] == 'No'):

            df.loc[i,'DTH']           = 0

            df.loc[i,'Cable']         = 0

            df.loc[i,'TVConnection']  = 0

            df.loc[i,'Monthly']       = 0

            df.loc[i,'Annually']      = 0

            df.loc[i,'Biannually']    = 0

            df.loc[i,'Annually']      = 0

            df.loc[i,'Cash']          = 0

            df.loc[i,'Bank transfer'] = 0

            df.loc[i,'Net Banking']   = 0

            df.loc[i,'Credit card']   = 0



        else:

            if(df.loc[i,'TVConnection'] == 'Cable'):

                df.loc[i,'Cable'] = 1

                df.loc[i,'DTH']   = 0

            else:

                df.loc[i,'Cable'] = 0

                df.loc[i,'DTH']   = 1

            df.loc[i,'TVConnection'] = 1



        if(df.loc[i,'Subscription'] == 'Monthly'):

            df.loc[i,'Subscription'] = 1

            df.loc[i,'Monthly']    = 1

            df.loc[i,'Annually']   = 0

            df.loc[i,'Biannually'] = 0

        elif(df.loc[i,'Subscription'] == 'Annually'):

            df.loc[i,'Subscription'] = 1

            df.loc[i,'Monthly']    = 0

            df.loc[i,'Annually']   = 1

            df.loc[i,'Biannually'] = 0

        elif(df.loc[i,'Subscription'] == 'Biannually'):

            df.loc[i,'Subscription'] = 1

            df.loc[i,'Monthly']    = 0

            df.loc[i,'Annually']   = 0

            df.loc[i,'Biannually'] = 1

        else:

            df.loc[i,'Subscription'] = 0

            df.loc[i,'Monthly']    = 0

            df.loc[i,'Annually']   = 0

            df.loc[i,'Biannually'] = 0



        if(df.loc[i,'PaymentMethod'] == 'Cash'):

            df.loc[i,'PaymentMethod'] = 1

            df.loc[i,'Cash']          = 1

            df.loc[i,'Bank transfer'] = 0

            df.loc[i,'Net Banking']   = 0

            df.loc[i,'Credit card']   = 0

        elif(df.loc[i,'PaymentMethod'] == 'Bank transfer'):

            df.loc[i,'PaymentMethod'] = 1

            df.loc[i,'Cash']          = 0

            df.loc[i,'Bank transfer'] = 1

            df.loc[i,'Net Banking']   = 0

            df.loc[i,'Credit card']   = 0

        elif(df.loc[i,'PaymentMethod'] == 'Net Banking'):

            df.loc[i,'PaymentMethod'] = 1

            df.loc[i,'Cash']          = 0

            df.loc[i,'Bank transfer'] = 0

            df.loc[i,'Net Banking']   = 1

            df.loc[i,'Credit card']   = 0

        elif(df.loc[i,'PaymentMethod'] == 'Credit card'):

            df.loc[i,'PaymentMethod'] = 1

            df.loc[i,'Cash']          = 0

            df.loc[i,'Bank transfer'] = 0

            df.loc[i,'Net Banking']   = 0

            df.loc[i,'Credit card']   = 1

        else:

            df.loc[i,'PaymentMethod'] = 0

            df.loc[i,'Cash']          = 0

            df.loc[i,'Bank transfer'] = 0

            df.loc[i,'Net Banking']   = 0

            df.loc[i,'Credit card']   = 0



    df['TotalCharges'] = df['TotalCharges'].apply(lambda x: fl(x))

    df = df.replace({'No':0,'Yes':1})

    df = df.replace({'Male':0,'Female':1})

    df = df.drop('Subscription',axis=1)

    df = df.drop('PaymentMethod',axis=1)

    print(df.info())

    return df



df = pd.read_csv('train.csv')

df = preprocess1(df)

df2 = pd.read_csv('test.csv')

df2 = preprocess1(df2)
colList = list(df.columns)

strong = [x for x in df.corr()['Satisfied'].index if np.abs(df.corr()['Satisfied'][x])>=0.2]

moderate = [x for x in df.corr()['Satisfied'].index if np.abs(df.corr()['Satisfied'][x])>=0.1 and np.abs(df.corr()['Satisfied'][x])<0.2]

weak = [x for x in df.corr()['Satisfied'].index if np.abs(df.corr()['Satisfied'][x])>=0 and np.abs(df.corr()['Satisfied'][x])<0.1]

print(strong)

print(moderate)

print(weak)

# Choosing only good columns

req_cols = df.columns

req_cols = req_cols.drop(weak)

# req_cols = req_cols.drop(moderate)

req_cols = req_cols.drop('TotalCharges')

req_cols = req_cols.drop('Satisfied')

# req_cols = req_cols.drop('custId')

x_train = df[req_cols].iloc[:]

y_train = df.loc[:,'Satisfied']

x_test  = df2[req_cols].iloc[:]



print("Total "+str(len(df.columns)))

print("Using "+str(len(x_train.columns)))

print("Test "+str(len(x_test.columns)))

# X_test  = df2[req_cols].iloc[:]
xtrain = x_train.values

ytrain = y_train

xtest  = x_test.values

x_train.columns
from imblearn.over_sampling import SMOTENC



# Oversampling for class balance

listCat = np.arange(0,17)

listCat = np.delete(listCat,[7,8])

smt = SMOTENC(categorical_features=listCat, random_state=42)

xover, yover = smt.fit_sample(x_train, y_train)
from imblearn.under_sampling import NearMiss

nm = NearMiss()

xunder, yunder = nm.fit_resample(x_train, y_train)

xunder.shape
from imblearn.under_sampling import InstanceHardnessThreshold

from sklearn.linear_model import LogisticRegression

iht = InstanceHardnessThreshold(random_state=0,estimator=LogisticRegression(solver='lbfgs', multi_class='auto'))

xunder, yunder = nm.fit_resample(x_train, y_train)
# result = np.concatenate((xunder,xtest),0)

from sklearn.preprocessing import StandardScaler 

ssc = StandardScaler()

# result[:,[7,8]] = ssc.fit_transform(result[:,[7,8]])

# xtest[:,[14,15]] = ssc.fit_transform(xtest[:,[14,15]])

# xunder[:,[14,15]] = ssc.fit_transform(xunder[:,[14,15]])

xtest[:,[7,8]] = ssc.fit_transform(xtest[:,[7,8]])

# xunder[:,[7,8]] = ssc.fit_transform(xunder[:,[7,8]])

# xover[:,[7,8]] = ssc.fit_transform(xover[:,[7,8]])

xtrain[:,[7,8]] = ssc.fit_transform(xtrain[:,[7,8]])
len(xunder)

xunder.shape

xtest.shape

# result.shape
from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from sklearn.neighbors import DistanceMetric

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# clustering = DBSCAN(eps=2, metric='euclidean', min_samples=250)

# clustering = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='cosine')

# clustering = SpectralClustering(n_clusters=2,affinity='rbf',assign_labels='discretize',random_state=42,gamma=0.0001)

clustering = LinearDiscriminantAnalysis()

clustering.fit(xtrain, ytrain) 

answer = clustering.predict(xtest)

# answer = clustering.fit_predict(result)

print(np.unique(answer))



# train_pred = answer[:4930]

# train_pred = answer[:2618]

# train_pred = answer[:7242]

# test_pred  = answer[-2113:]
from sklearn.metrics import precision_score, recall_score, accuracy_score, auc, roc_curve



y = yunder

pred = train_pred



print("Precision = {}".format(precision_score(y, pred, average='macro')))

print("Recall = {}".format(recall_score(y, pred, average='macro')))

print("Accuracy = {}".format(accuracy_score(y, pred)))

fpr, tpr, thresholds = roc_curve(y, pred)

print("AUC = {}".format(auc(fpr, tpr)))
finalA = answer[-2113:]

answerFrame = pd.DataFrame({'custId':df2['custId'], 'Satisfied':finalA})

answerFrame.to_csv('submission4.csv',index=None)
# np.unique(f,return_counts=True)
# plt.plot(df['TVConnection'],df['Subscription'])

# plt.figure(figsize=(20,20))

# plt.scatter(df[df['Satisfied']==1]['TVConnection'],df[df['Satisfied']==1]['Subscription'],c=df[df['Satisfied']==1]['Satisfied'],alpha=0.1,marker='x')

# plt.scatter(df[df['Satisfied']==0]['TVConnection'],df[df['Satisfied']==0]['Subscription'],c=df[df['Satisfied']==0]['Satisfied'],alpha=0.1,marker='o')

# plt.show()

# for i in range(0,2):

#     for j in range(0,2):

#             print("Satisfied 0 : "+str(i)+" "+str(j)+" "+str(len(df[(df['Satisfied']==0) & (df['TVConnection']==i) & (df['Subscription']==j)])))



# for i in range(0,2):

#     for j in range(0,2):

#             print("Satisfied 1 : "+str(i)+" "+str(j)+" "+str(len(df[(df['Satisfied']==1) & (df['TVConnection']==i) & (df['Subscription']==j)])))

            

# plt.hist(df['Satisfied'])

# plt.hist(df[df['Satisfied']==1]['TVConnection'])

# plt.show()

# ax = sns.swarmplot(x='Satisfied', y='MonthlyCharges',data=df)



# plt.show()

# from sklearn.manifold import TSNE

# X_embedded = TSNE(n_components=2).fit_transform(x_train)

# # plt.scatter()

# plt.scatter(x=X_embedded[:,0], y=X_embedded[:,1], c=df['Satisfied'])
# # Change categorical to numbers to get correlation

# cat_columns = df.select_dtypes(['object']).columns

# df[cat_columns] = df[cat_columns].astype('category')

# df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)



# # Remove low correlation columns

# corr = df.corr()

# good_corr_col = [i for i in range(0,20) if np.abs(corr.iloc[i][20]) > 0.1]

# print(req_cols)
for i in df.columns:

    print(i+" "+str(df[i].unique())+" "+str(df[i].nunique()))
## AGGLOMERATE WITH XUNDER GIVES 0.59

## LDA WITH XUNDER gives 0.72

df = pd.read_csv('train.csv')

df2 = pd.read_csv('test.csv')

len(df2)
#Data Manipulation

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



def dataManip(data):

    # Taking care of TotalCharges

    data['TotalCharges'] = data["TotalCharges"].replace(" ",0)

    data = data.reset_index()[data.columns]

    data["TotalCharges"] = data["TotalCharges"].astype(float)



    # Fix Channels

    replace_cols = [ 'Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6']

    for i in replace_cols : 

        data[i]  = data[i].replace({'No tv connection' : 'No'})



    # Fix Channels

    replace_cols = ['HighSpeed']

    for i in replace_cols : 

        data[i]  = data[i].replace({'No internet' : 'No'})



    # Fix Senior Cit

    data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})



    #Tenure to categorical column

    def tenure_lab(data) :

        if data["tenure"] <= 12 :

            return "Tenure_0-12"

        elif (data["tenure"] > 12) & (data["tenure"] <= 24 ):

            return "Tenure_12-24"

        elif (data["tenure"] > 24) & (data["tenure"] <= 48) :

            return "Tenure_24-48"

        elif (data["tenure"] > 48) & (data["tenure"] <= 60) :

            return "Tenure_48-60"

        elif data["tenure"] > 60 :

            return "Tenure_gt_60"

    data["tenure_group"] = data.apply(lambda data:tenure_lab(data),

                                          axis = 1)



    #Separating catagorical and numerical columns

    Id_col     = ['custId']

    tar_col    = ['Satisfied']

    cat_cols   = data.nunique()[data.nunique() < 6].keys().tolist()

    num_cols   = [x for x in data.columns if x not in cat_cols + Id_col + tar_col]



    #Binary columns with 2 values

    bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()

    print("Binary : "+str(bin_cols))

    #Columns more than 2 values

    multi_cols = [i for i in cat_cols if i not in bin_cols]

    print("Multi  : "+str(multi_cols))

    

    #Label encoding Binary columns

    le = LabelEncoder()

    for i in bin_cols :

        data[i] = le.fit_transform(data[i])

    

    #Duplicating columns for multi value columns

    data = pd.get_dummies(data = data,columns = multi_cols )



    #Scaling Numerical columns

    std = StandardScaler()

    scaled = std.fit_transform(data[num_cols])

    scaled = pd.DataFrame(scaled,columns=num_cols)



    #dropping original values merging scaled values for numerical columns

    df_data_og = data.copy()

    data = data.drop(columns = num_cols,axis = 1)

    data = data.merge(scaled,left_index=True,right_index=True,how = "left")

    

    return data

    

df = dataManip(df)

df2 = dataManip(df2)

print(len(df2))
print(df.head().columns)

Id_col     = ['custId']

tar_col    = ['Satisfied']

cat_cols   = df.nunique()[df.nunique() < 6].keys().tolist()

cat_cols   = [x for x in df.columns if x not in Id_col + tar_col]

num_cols   = [x for x in df.columns if x not in cat_cols + Id_col + tar_col]
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split



cols    = [i for i in df.columns if i not in Id_col+tar_col]

smote_X = df[cols]

smote_Y = df[tar_col]

#Split train and test df

smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,test_size = .25 ,random_state = 111)



#oversampling minority class using smote

os = SMOTE(random_state = 0)

os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)

os_smote_X = pd.DataFrame(data = os_smote_X,columns=cols)

os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=tar_col)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import precision_score, recall_score, accuracy_score, auc, roc_curve



clustering = LinearDiscriminantAnalysis()

clustering.fit(os_smote_X, os_smote_Y) 

answer = clustering.predict(smote_test_X)



print("Precision = {}".format(precision_score(smote_test_Y, answer, average='macro')))

print("Recall = {}".format(recall_score(smote_test_Y, answer, average='macro')))

print("Accuracy = {}".format(accuracy_score(smote_test_Y, answer)))

fpr, tpr, thresholds = roc_curve(smote_test_Y, answer)

print("AUC = {}".format(auc(fpr, tpr)))
test_x = df2[cols]

answer = clustering.predict(test_x)

finalA = answer[-2113:]

answerFrame = pd.DataFrame({'custId':df2['custId'], 'Satisfied':finalA})

answerFrame.to_csv('submission4.csv',index=None)
len(df2)