
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import pydicom

import warnings
warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

train_data.head()
print(train_data.shape)
print('----------------------------')

#null values in test & train data
print(train_data.isnull().sum())
print('----------------------------')
print(test_data.isnull().sum())
print('----------------------------')

#data type of each column
print(train_data.dtypes)
#total unique id's of the patients as from shape we know total id's are 1549
train_data['Patient'].nunique()
fig, ax = plt.subplots(2,2, figsize=(15,7))
plt.subplots_adjust(bottom=0, right=1, top=1)

ax[0,0].set_title('Total No.', color='red', fontsize = 15)
sns.countplot('Sex', data= train_data, palette='Blues', ax= ax[0,0] )

ax[1,0].set_title('Smoking Status', color='red', fontsize = 15)
sns.countplot('SmokingStatus', data=train_data, palette='Blues', ax=ax[1,0])

ax[1,1].set_title('Smoking status by Sex', color='red', fontsize = 15)
sns.countplot('Sex', data=train_data, hue='SmokingStatus', palette = 'Blues', ax=ax[1,1])

#gender percentage
count = [train_data['Sex'].value_counts().values]
labels = ['male', 'female']
explode = (0.1,0)

ax[0,1].set_title('Percentage of male and female', color='red', fontsize = 15)
ax[0,1].pie(count, labels=labels,
       explode = explode, 
       shadow= True,  
       autopct='%1.1f%%')
ax[0,1].axis('equal') 
plt.show()


id_count = pd.DataFrame(train_data['Patient'].value_counts())
id_count['id'] =id_count.index
id_count.columns= ['count','id']

fig = px.bar(id_count, x='id',y ='count',color='count')
fig.update_xaxes(showticklabels=False)
fig.show()
fig, ax = plt.subplots(1,4, figsize =(23,3))
plt.subplots_adjust(bottom=0, right=1, top=1)

ax[0].set_title('Age', color='red', fontsize = 15)
sns.distplot(train_data['Age'],  color="r", ax=ax[0])

ax[1].set_title('Weeks', color='red', fontsize = 15)
sns.distplot(train_data['Weeks'],  color="g", ax=ax[1])

ax[2].set_title('Percent', color='red', fontsize = 15)
sns.distplot(train_data['Percent'],  color="c", ax=ax[2])

ax[3].set_title('FVC', color='red', fontsize = 15)
sns.distplot(train_data['FVC'],  color="b", ax=ax[3])
plt.show()
fig = px.histogram(train_data, x='Age', color='SmokingStatus', marginal="violin",color_discrete_map={'Ex-smoker':'#393E46','Never smoked':'#7c7c79','Currently smokes':'#04d6cb'})
fig.update_traces(marker_line_color='cyan',marker_line_width=1.5, opacity=0.85)
fig.show()
fig = px.histogram(train_data, x='Age', color='Sex',marginal="violin", color_discrete_map={'Male':'#393E46','Female':'#04d6cb'})
fig.update_traces(marker_line_color='black',marker_line_width=1.5, opacity=0.85)
fig.show()
fig = px.histogram(train_data, x='FVC', color='Sex', title='FVC by Gender', marginal="violin", color_discrete_map={'Male':'#393E46','Female':'#04d6cb'})
fig.update_traces(marker_line_color='black',marker_line_width=1.5, opacity=0.85)
fig.show()
fig = px.histogram(train_data, x='FVC', color='SmokingStatus', marginal="violin",color_discrete_map={'Ex-smoker':'#393E46','Never smoked':'#7c7c79','Currently smokes':'#04d6cb'})
#fig.update_traces(marker_line_color='black',marker_line_width=1.5, opacity=0.85)
fig.update_layout(title='FVC by Smoking Status')
fig.show()
img_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00027637202179689871102/125.dcm"
img = pydicom.dcmread(img_path)
fig, ax = plt.subplots(1,3, figsize=(20,10))
ax[0].imshow(img.pixel_array,cmap='nipy_spectral')
ax[1].imshow(img.pixel_array,cmap='hot')
ax[2].imshow(img.pixel_array,cmap=plt.cm.bone)

plt.show()