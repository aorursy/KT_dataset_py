import pandas as pd

from sklearn.cluster import KMeans

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
data = pd.read_csv('https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/cases.csv')

data.head()
data.shape
data_imp = data[['age', 'sex', 'health_region', 'province', 'report_week', 'travel_yn', 'travel_history_country']]

data_imp.tail()
data_imp.isnull().sum().sort_values(ascending = False)
data_imp = data_imp.fillna('None')
choose = data_imp[(data_imp['age'] == 'Not Reported') & (data_imp['sex'] == 'Not Reported') & (data_imp['travel_history_country'] == 'Not Reported')]

index = choose.index.tolist()

data_imp.drop(index, inplace = True)

data_imp.head()
label_dic = {}

encode = LabelEncoder()

for feature in ['age', 'sex', 'health_region', 'province', 'report_week', 'travel_yn', 'travel_history_country']:

    data_imp[feature] = encode.fit_transform(data_imp[feature])

    keys = encode.classes_

    values = encode.transform(encode.classes_)

    dictionary = dict(zip(keys, values))

    label_dic[feature] = dictionary
data_imp.head()
kmeans = KMeans(n_clusters=3, random_state=0).fit(data_imp)

result = kmeans.predict(data_imp)

result
k = kmeans.predict([[100,100,100,100,100,100,100]])

k
if 1 in result.tolist():

    print('yes')