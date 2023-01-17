import pandas as pd

import numpy as np



from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import classification_report, f1_score, accuracy_score



import matplotlib.pyplot as plt

import seaborn as sns

train_df = pd.read_csv("../input/data-mining-assignment-2/train.csv")

test_df = pd.read_csv("../input/data-mining-assignment-2/test.csv")

label_df = train_df['Class']

train_df = train_df.drop(['ID', 'Class'], axis=1)

sub_df = pd.DataFrame(test_df['ID'], columns=['ID'])

test_df = test_df.drop(['ID'], axis=1)
metal_dict = {

    'Silver': 1,

    'Gold': 2,

    'Diamond': 4,

    'Platinum':3

}



boolean_dict = {

    'Yes': 1,

    'No': 0

}



gender_dict = {

    'Male': 0,

    'Female': 1

}



level_dict = {

    'Low': 1,

    'Medium': 2,

    'High': 3

}
def categorical_to_num(col, cat_dict):

    return [cat_dict[val] for val in col]
def preprocess_data(df):

    

    df['col2'] = categorical_to_num(df['col2'], metal_dict)

    df['col11'] = categorical_to_num(df['col11'], boolean_dict)

    df['col37'] = categorical_to_num(df['col37'], gender_dict)

    df['col44'] = categorical_to_num(df['col44'], boolean_dict)

    df['col56'] = categorical_to_num(df['col56'], level_dict)

    return df
train_df = preprocess_data(train_df)

test_df = preprocess_data(test_df)
def correlation_correction(df,threshold):

    corr_matrix = df.corr().abs()



    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



    # Find features with correlation greater than 0.95

    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    return to_drop
correlated_cols = correlation_correction(train_df,0.99)
correlation_matrix = train_df.corr()

correlation_matrix.loc[correlated_cols,correlated_cols]
fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(correlation_matrix.loc[correlated_cols,correlated_cols], annot=True)

plt.show()
comp_corr_cols = ['col45','col46']

train_df.drop(comp_corr_cols,axis=1,inplace=True)

test_df.drop(comp_corr_cols,axis=1,inplace=True)
display(train_df.head())

display(test_df.head())
categorical_col = ['col2','col11','col37','col44','col56']



numerical_col = list(set(train_df.columns) - set(categorical_col))
# encoder = OneHotEncoder()

# encoder.fit(pd.concat([train_df[categorical_col], test_df[categorical_col]], axis=0))



scaler = StandardScaler()

scaler.fit(train_df.astype('float64'))
def get_scaled_df(df, scaler):

    scaled_df = pd.DataFrame(scaler.transform(df.astype('float64')))

    scaled_df.columns = df.columns

    return scaled_df
scaled_train_df = get_scaled_df(train_df, scaler)

scaled_test_df = get_scaled_df(test_df, scaler)
display(scaled_train_df.head())

display(scaled_test_df.head())
from sklearn.manifold import Isomap



embedding = Isomap(n_components=2)

train_x_iso = embedding.fit_transform(scaled_train_df)

test_x_iso = embedding.transform(scaled_test_df)
x1 = [train_x_iso[i][0] for i in range(len(train_x_iso))]

y1 = [train_x_iso[i][1] for i in range(len(train_x_iso))]



x2 = [test_x_iso[i][0] for i in range(len(test_x_iso))]

y2 = [test_x_iso[i][1] for i in range(len(test_x_iso))]
plt.figure(figsize=(20,20))

fig, ax = plt.subplots()

sc1 = ax.scatter(x1,y1,c="red",label = "Training Data")

sc1 = ax.scatter(x2,y2,c="blue",label = "Testing Data")

ax.legend()

plt.title('Testing and training data reduced to 2 dimensions to look for dissimilarity')

# plt.savefig('Isomap')
from sklearn.metrics import roc_auc_score

drop_list = []

drifts = []

cols = []

for col in train_df.columns:

    # Select column

    X_train = pd.DataFrame(train_df.loc[:,col])

    X_test = pd.DataFrame(test_df.loc[:,col])



    # Add origin feature

    X_train["target"] = 0

    X_test["target"] = 1



    # Merge datasets

    X_tmp = pd.concat((X_train, X_test),

                      ignore_index=True).drop(['target'], axis=1)

    y_tmp= pd.concat((X_train.target, X_test.target),

                   ignore_index=True)



    X_train_tmp, X_test_tmp, \

    y_train_tmp, y_test_tmp = train_test_split(X_tmp,

                                               y_tmp,

                                               test_size=0.25,

                                               random_state=1)



    # Use Random Forest classifier

    rf = RandomForestClassifier(n_estimators=50,

                                n_jobs=-1,

                                max_features=1.,

                                min_samples_leaf=5,

                                max_depth=5,

                                random_state=1)



    # Fit 

    rf.fit(X_train_tmp, y_train_tmp)



    # Predict

    y_pred_tmp = rf.predict_proba(X_test_tmp)[:, 1]



    # Calculate ROC-AUC

    score = roc_auc_score(y_test_tmp, y_pred_tmp)

#     cols.append(col)

    if(np.mean(score)>0.8):

        drop_list.append(col)

        print(col,np.mean(score))

#     drifts.append((max(np.mean(score), 1 - np.mean(score)) - 0.5) * 2)

#     print(cols)

#     print(drifts)
drop_list
rf = RandomForestClassifier(n_estimators=200, max_depth=9,max_features=10,random_state=52)

rf.fit(scaled_train_df, label_df)

### plotting importances

features = scaled_train_df.columns.values

imp = rf.feature_importances_

indices = np.argsort(imp)[::-1][:8]



#plot

plt.figure(figsize=(8,5))

plt.title("Most Relevent features for classification")

plt.bar(range(len(indices)), imp[indices], align='center')

plt.xticks(range(len(indices)), features[indices], rotation='vertical')

plt.xlim([-1,len(indices)])

plt.savefig('Important Features')

plt.show()
lst = [value for value in features[indices] if value in  drop_list]

drop_list_final = [value for value in drop_list if value not in lst]

drop_list_final
scaled_train_df.drop(drop_list_final,axis=1,inplace=True)

scaled_test_df.drop(drop_list_final,axis=1,inplace=True)
display(scaled_train_df.head())

display(scaled_test_df.head())
clf = RandomForestClassifier(n_estimators=1000, max_depth=6,max_features='log2', n_jobs=-1, random_state=52,class_weight="balanced")

clf.fit(scaled_train_df, label_df)

pred = clf.predict(scaled_test_df)
pd.DataFrame(pred)[0].value_counts()
sub_df['Class'] = pred
sub_df.head()
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub_df)
sub_df.to_csv('2016B4A70275G_final.csv', index=False)