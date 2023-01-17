# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
data=pd.read_csv('../input/train_LZdllcl.csv')
data.info()
#Education and Previous year rating has missing values
#Try to find out the type of missing values in previous year rating type
data.previous_year_rating.value_counts()
#Try to find out the relation between prev year rating and length_of_service
#No of missing value in prev year rating
def impute_missing_val(data):
    print('No of missing value in prev year rating=',len(data[data.previous_year_rating.isnull()].
                                                     previous_year_rating))
    data_null_re=data[['previous_year_rating','length_of_service']]
    print(data_null_re[data_null_re.previous_year_rating.isnull()].head(5))
    data_null_re[data_null_re.previous_year_rating.isnull()].groupby(['length_of_service']).count()
#CREATE A COLUMN AFTER RPLACING THE MISSING VALUE
    data_null_re.fillna(0,inplace=True)
    print(data_null_re.head(11))
#Count the missinng values in education column
    print("No. of missing values in education column",data.education.isnull().sum())
#Check the proportion of differnt education type
    print(data.education.value_counts())
    data_edu_null_re=data[['education']]
    data_edu_null_re.fillna("""Bachelor's""",inplace=True)
    data['education']=data_edu_null_re
    data['previous_year_rating']=data_null_re[['previous_year_rating']]
    #Drop the  employee_id column
    data.drop(['employee_id'],axis=1,inplace=True)
    return data
#Now you can see that none of the column having miissing value
data=impute_missing_val(data)
data.info()
data.head()
#Try to have some insights on the number of levels for each categorical feature
cat_features=['department','region','education','gender','recruitment_channel','previous_year_rating','KPIs_met >80%','awards_won?']
for column in data[cat_features]:
    #print(set(data[[column]]))
    print('Unique levels in {0} column'.format(str(column)),set(data[column]))
print('No. of unique levels in region column',len(list(set(data['region']))))
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
from sklearn.preprocessing import OneHotEncoder
data_onehot_enc=data

onehot_encoder=OneHotEncoder()
colms=['department','region','education','recruitment_channel']
onehot_encoded=onehot_encoder.fit_transform(data_onehot_enc[colms])
data_onehot_enc.drop(colms,inplace=True,axis=1)
#Change the encoded feature into df

df_onehot_enc=pd.DataFrame(data=onehot_encoded.toarray(),columns=onehot_encoder.get_feature_names())
#Covert Gender column using Label Encoder
from sklearn.preprocessing import LabelEncoder
label_encoded=LabelEncoder().fit_transform(data_onehot_enc['gender'])
data_onehot_enc['gender']=label_encoded

df_onehot_enc=pd.concat([data_onehot_enc,df_onehot_enc],axis=1).info()
data=pd.read_csv('../input/train_LZdllcl.csv')
data_label_enc=impute_missing_val(data)
le_enc=LabelEncoder()
colms=['gender','department','region','education','recruitment_channel']
data_label_enc['gender']=le_enc.fit_transform(data_label_enc['gender'])
data_label_enc['department']=le_enc.fit_transform(data_label_enc['department'])
print(le_enc.classes_)
data_label_enc['region']=le_enc.fit_transform(data_label_enc['region'])
data_label_enc['education']=le_enc.fit_transform(data_label_enc['education'])
data_label_enc['recruitment_channel']=le_enc.fit_transform(data_label_enc['recruitment_channel'])
data_label_enc.head()
data=pd.read_csv('../input/train_LZdllcl.csv')
data_binary_enc=impute_missing_val(data)
data_binary_enc.department.value_counts()
data_label_enc.department.value_counts()
import category_encoders as ce
lb=ce.BinaryEncoder()
binarized_dept=lb.fit_transform(data_binary_enc[['department']])
lb=ce.BinaryEncoder()
binarized_region=lb.fit_transform(data_binary_enc[['region']])
lb=ce.BinaryEncoder()
binarized_gender=lb.fit_transform(data_binary_enc[['gender']])
lb=ce.BinaryEncoder()
binarized_channel=lb.fit_transform(data_binary_enc[['recruitment_channel']])
lb=ce.BinaryEncoder()
binarized_edu=lb.fit_transform(data_binary_enc[['education']])

binarized_dept.describe()
#You can see that after binary encoding region_0 and department_0 column is redindant
binarized_region.drop(['region_0'],axis=1,inplace=True)
binarized_dept.drop(['department_0'],axis=1,inplace=True)
binarized_gender.drop(['gender_0'],axis=1,inplace=True)
binarized_edu.drop(['education_0'],axis=1,inplace=True)
binarized_channel.drop(['recruitment_channel_0'],axis=1,inplace=True)
data_binary_enc.drop(['gender','department','region','education','recruitment_channel'],axis=1,inplace=True)
data_binary_enc=pd.concat([binarized_region,binarized_dept,binarized_gender,binarized_edu,binarized_channel,data_binary_enc],axis=1)
data_binary_enc.info()
data=pd.read_csv('../input/train_LZdllcl.csv')
df_hash_enc_data=impute_missing_val(data)

ce_hash=ce.HashingEncoder(cols=['department','region','education','recruitment_channel'],
                          drop_invariant=True,n_components=10,return_df=True,hash_method='md5')
df_hash=ce_hash.fit_transform(df_hash_enc_data[['department','region','education','recruitment_channel']])
df_hash.describe()
df_hash_enc_data.drop(['department','region','education','recruitment_channel'],axis=1,inplace=True)
le=LabelEncoder()
df_hash_enc_data['gender']=le.fit_transform(df_hash_enc_data['gender'])
df_hash_enc_data=pd.concat([df_hash,df_hash_enc_data],axis=1)
df_hash_enc_data.info()