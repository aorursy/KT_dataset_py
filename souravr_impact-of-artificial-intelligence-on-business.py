import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import scipy.stats as ss
data = pd.read_csv("/kaggle/input/dataset-responses-ai/Dataset_of_AI_Responses.csv", encoding='latin1')
data = data.iloc[:,1:]
data['Google maps  GM'].replace({"Netural": "Neutral"}, inplace=True)
data_cat = data
data = data.astype(str)
data.head()

print(data.dtypes)
import category_encoders as ce

# Get a new clean dataframe
obj_df = data.select_dtypes(include=['object']).copy()

# Specify the columns to encode then fit and transform
encoder = ce.backward_difference.BackwardDifferenceEncoder()
encoder.fit(obj_df, verbose=1)

# Only display the first 8 columns for brevity
data = encoder.transform(obj_df)
data = data.iloc[:, 1:]
# # instantiate labelencoder object
# le = LabelEncoder()
# # Categorical boolean mask
# categorical_feature_mask = data.dtypes==object# filter categorical columns using mask and turn it into a list
# categorical_cols = data.columns[categorical_feature_mask].tolist()
# # apply le on categorical feature columns
# data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
# data.head(10)
# #Correlation plot of the google application variables 
# corr = data.iloc[:,5:10].corr()
# corr.style.background_gradient(cmap='coolwarm')

# #Correlation plot of the AI application variables 
# corr = data.iloc[:,10:].corr()
# corr.style.background_gradient(cmap='coolwarm')
data.columns
data_cat.columns
#Frequency table
pd.crosstab(data_cat['A1'],data_cat['Google Smart replies  GSR'])
print(chi2_contingency(pd.crosstab(data_cat['A1'],data_cat['Google Smart replies  GSR'])))
#Chisuare results sig/non sig for the nominal variable gender  G1
p_value = []
x = []
y = []
significant = []
for i in range(0,1,1):
    for j in range(6,len(data_cat.columns),1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("Significant")
        else:
            significant.append("Not Significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
AI = {'Independent variable': x, 'Response variable': y, 'P_value' : p_value, 'Relationship': significant}
AI = pd.DataFrame.from_dict(AI)
AI
#Chisuare results sig/non sig for the nominal variable age A1
p_value = []
x = []
y = []
significant = []
for i in range(1,2,1):
    for j in range(6,len(data_cat.columns),1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("Significant")
        else:
            significant.append("Not Significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
AI = {'Independent variable': x, 'Response variable': y, 'P_value' : p_value, 'Relationship': significant}
AI = pd.DataFrame.from_dict(AI)
AI
#Chisuare results sig/non sig for the nominal variable married  M1
p_value = []
x = []
y = []
significant = []
for i in range(2,3,1):
    for j in range(6,len(data_cat.columns),1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("Significant")
        else:
            significant.append("Not Significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
AI = {'Independent variable': x, 'Response variable': y, 'P_value' : p_value, 'Relationship': significant}
AI = pd.DataFrame.from_dict(AI)
AI
# Chisuare results sig/non sig for the nominal variable employment  E1
p_value = []
x = []
y = []
significant = []
for i in range(3,4,1):
    for j in range(6,len(data_cat.columns),1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("Significant")
        else:
            significant.append("Not Significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
AI = {'Independent variable': x, 'Response variable': y, 'P_value' : p_value, 'Relationship': significant}
AI = pd.DataFrame.from_dict(AI)
AI
#Chisuare results sig/non sig for the nominal variable degree  Q1
p_value = []
x = []
y = []
significant = []
for i in range(4,5,1):
    for j in range(6,len(data_cat.columns),1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("Significant")
        else:
            significant.append("Not Significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
AI = {'Independent variable': x, 'Response variable': y, 'P_value' : p_value, 'Relationship': significant}
AI = pd.DataFrame.from_dict(AI)
AI
#Relationship among the google application variables 
p_value = []
x = []
y = []
significant = []
for i in range(10,len(data_cat.columns)-1,1):
    for j in range(i+1,len(data_cat.columns),1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("significant")
        else:
            significant.append("not significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
AI = {'attr1': x, 'attr2': y, 'P_value' : p_value, 'Relationship': significant}
AI = pd.DataFrame.from_dict(AI)
AI
p_value = []
x = []
y = []
significant = []
for i in range(0,len(data_cat.columns)-1,1):
    for j in range(i+1,len(data_cat.columns),1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("significant")
        else:
            significant.append("not significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
AI = {'attr1': x, 'attr2': y, 'P_value' : p_value, 'Relationship': significant}
AI = pd.DataFrame.from_dict(AI)
AI[AI['Relationship']=="significant"][:30]
#Relationship among AI application variables  
p_value = []
x = []
y = []
significant = []
for i in range(5,9,1):
    for j in range(i+1,10,1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("significant")
        else:
            significant.append("not significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
google = {'attr1': x, 'attr2': y, 'P_value' : p_value, 'Relationship': significant}
google = pd.DataFrame.from_dict(google)
google
p_value = []
x = []
y = []
significant = []
for i in range(0,9,1):
    for j in range(i+1,10,1):
        p_value.append(chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1])
        if chi2_contingency(pd.crosstab(data_cat.iloc[:,i],data_cat.iloc[:,j]))[1]<=0.005:
            significant.append("significant")
        else:
            significant.append("not significant")
        x.append(data_cat.columns[i])
        y.append(data_cat.columns[j])
        
google = {'attr1': x, 'attr2': y, 'P_value' : p_value, 'Relationship': significant}
google = pd.DataFrame.from_dict(google)
google[google['Relationship']=="significant"]
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
data.columns
normalized_df = data.iloc[:,21:44]
normalized_df = pd.DataFrame(normalized_df) 
# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(data.iloc[:,20:44]) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df) 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
  
X_principal.head(2)
plt.figure(figsize =(6, 6)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward'))) 


silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(X_principal, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(X_principal))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 


hc = AgglomerativeClustering(n_clusters=3,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X_principal)
y_hc
# Visualizing the clustering 
plt.figure(figsize = (10,10))
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = AgglomerativeClustering(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show() 
new_data = data_cat.iloc[:,:10]
new_data['cluster'] = y_hc
new_data.columns
print(new_data[new_data['cluster']==2]['Google maps  GM'].value_counts())
print(new_data[new_data['cluster']==2]['Google Smart replies  GSR'].value_counts())
print(new_data[new_data['cluster']==2]['Google search query  GSQ '].value_counts())
print(new_data[new_data['cluster']==2]['Google page  rank   GPR           '].value_counts())
print(new_data[new_data['cluster']==2]['Google Email filter   GEF  '].value_counts())
print(new_data[new_data['cluster']==2]['G1'].value_counts())
print(new_data[new_data['cluster']==2]['A1'].value_counts())
print(new_data[new_data['cluster']==2]['M1'].value_counts())
print(new_data[new_data['cluster']==2]['E1'].value_counts())
print(new_data[new_data['cluster']==2]['Q1'].value_counts())
print(new_data[new_data['cluster']==1]['Google maps  GM'].value_counts())
print(new_data[new_data['cluster']==1]['Google Smart replies  GSR'].value_counts())
print(new_data[new_data['cluster']==1]['Google search query  GSQ '].value_counts())
print(new_data[new_data['cluster']==1]['Google page  rank   GPR           '].value_counts())
print(new_data[new_data['cluster']==1]['Google Email filter   GEF  '].value_counts())
print(new_data[new_data['cluster']==1]['G1'].value_counts())
print(new_data[new_data['cluster']==1]['A1'].value_counts())
print(new_data[new_data['cluster']==1]['M1'].value_counts())
print(new_data[new_data['cluster']==1]['E1'].value_counts())
print(new_data[new_data['cluster']==1]['Q1'].value_counts())
print(new_data[new_data['cluster']==0]['Google maps  GM'].value_counts())
print(new_data[new_data['cluster']==0]['Google Smart replies  GSR'].value_counts())
print(new_data[new_data['cluster']==0]['Google search query  GSQ '].value_counts())
print(new_data[new_data['cluster']==0]['Google page  rank   GPR           '].value_counts())
print(new_data[new_data['cluster']==0]['Google Email filter   GEF  '].value_counts())
print(new_data[new_data['cluster']==0]['G1'].value_counts())
print(new_data[new_data['cluster']==0]['A1'].value_counts())
print(new_data[new_data['cluster']==0]['M1'].value_counts())
print(new_data[new_data['cluster']==0]['E1'].value_counts())
print(new_data[new_data['cluster']==0]['Q1'].value_counts())
# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(data.iloc[:,45:]) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df) 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
  
X_principal.head(2)
plt.figure(figsize =(6, 6)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward'))) 
silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(X_principal, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(X_principal))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 



hc = AgglomerativeClustering(n_clusters=2,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X_principal)
# Visualizing the clustering 
plt.figure(figsize = (10,10))
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = AgglomerativeClustering(n_clusters = 2).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show() 
new_data = data_cat
new_data['cluster'] = y_hc
new_data.columns
print(new_data[new_data['cluster']==0]['G1'].value_counts())
print(new_data[new_data['cluster']==0]['A1'].value_counts())
print(new_data[new_data['cluster']==0]['M1'].value_counts())
print(new_data[new_data['cluster']==0]['E1'].value_counts())
print(new_data[new_data['cluster']==0]['Q1'].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,10].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,11].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,12].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,13].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,14].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,15].value_counts())
print(new_data[new_data['cluster']==1]['G1'].value_counts())
print(new_data[new_data['cluster']==1]['A1'].value_counts())
print(new_data[new_data['cluster']==1]['M1'].value_counts())
print(new_data[new_data['cluster']==1]['E1'].value_counts())
print(new_data[new_data['cluster']==1]['Q1'].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,10].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,11].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,12].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,13].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,14].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,15].value_counts())
# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(data.iloc[:,:20]) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df) 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
  
X_principal.head(2)
plt.figure(figsize =(6, 6)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward'))) 
silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(X_principal, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(X_principal))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 

hc = AgglomerativeClustering(n_clusters=3,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X_principal)
# Visualizing the clustering 
plt.figure(figsize = (10,10))
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = AgglomerativeClustering(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show() 
new_data = data_cat.iloc[:,:5]
new_data['cluster'] = y_hc
new_data
print(new_data[new_data['cluster']==0].iloc[:,0].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,1].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,2].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,3].value_counts())
print(new_data[new_data['cluster']==0].iloc[:,4].value_counts())

print(new_data[new_data['cluster']==1].iloc[:,0].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,1].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,2].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,3].value_counts())
print(new_data[new_data['cluster']==1].iloc[:,4].value_counts())

print(new_data[new_data['cluster']==2].iloc[:,0].value_counts())
print(new_data[new_data['cluster']==2].iloc[:,1].value_counts())
print(new_data[new_data['cluster']==2].iloc[:,2].value_counts())
print(new_data[new_data['cluster']==2].iloc[:,3].value_counts())
print(new_data[new_data['cluster']==2].iloc[:,4].value_counts())
