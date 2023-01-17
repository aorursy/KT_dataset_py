# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
# Import Data
data_import = pd.read_csv('../input/dmassign1/data.csv').set_index('ID')

# Preview Data
data_import.tail()

# Divide into input and class labels
data_input = data_import.loc[:, 'Col1':'Col197']
class_labels = pd.DataFrame(data_import.loc[:, 'Class'])

class_labels.head()
# Simple Summary
data_input.info()
# How many columns are determined as 'object' type?
cat_data = data_input.select_dtypes(include=['object']).copy()
# # Preview
# cat_data.head(30)
# Replace ? with NaN in the numerical columns 
# Both start and stop of .loc slice included
data_input.loc[:, 'Col30':'Col197'] = data_input.loc[:, 'Col30':'Col197'].replace('?', np.NaN)

# Convert all numerical columns to floats as necessary
data_input.loc[:, 'Col1':'Col187'] = data_input.loc[:, 'Col1':'Col187'].astype('float64') 

# How many columns are determined as 'object' type now
cat_data = data_input.select_dtypes(include=['object']).copy()

#cat_data.head(20)
# How many rows with >1 null values exist
cat_data[cat_data.isnull().any(axis=1)]

# Fill NA of categorical data with mode, convert to category type, one-hot encode these columns

cat_columns = cat_data.columns # list of categorical columns
mode_list = data_input.filter(cat_columns).mode() #list of modes

data_input[cat_columns] = data_input[cat_columns].fillna(mode_list.iloc[0]) # filling NaN
data_input.loc[:, 'Col189':'Col197'] = data_input.loc[:, 'Col189':'Col197'].astype('category')

# One-hot encode
data_input_onehot = pd.get_dummies(data_input, drop_first=True)

# How many rows contain NaN values now
data_input_onehot[data_input_onehot.isnull().any(axis=1)].shape
# Fill any remaining NaNs - which are only the numerical attributes - with mean now
data_input_onehot = data_input_onehot.fillna(data_input_onehot.mean())

data_input_onehot.head()
data_input_onehot.head()
data_input_onehot.info()

# Broader Box-Plot For the Data
plt.figure(figsize=(30, 10))
data_input_onehot.boxplot(column=list(data_input_onehot.columns)[:-32])
plt.title("Coarse Box-Plot of Numerical Attributes", size=18)
plt.xticks([])
plt.savefig('2016A7TS0719G_fig1.png', dpi=350)

# Zoom in on Box plots
plt.figure(figsize=(30, 10))
data_input_onehot.boxplot(column=list(data_input_onehot.columns)[-70:-32])

plt.title("Zoomed-in Box-Plot of Numerical Attributes", size=18)
plt.savefig('2016A7TS0719G_fig2.png', dpi=350)
# Check distribution of features in the range where box-plots look very similar
sns.distplot(data_input_onehot['Col160'])
plt.title("Visualizing Distribution of Randomly Chosen Feature 'Col160'", size=10)
plt.savefig('2016A7TS0719G_fig3.png', dpi=350)


# Feature Clipping on all Numerical Columns till 178 - region where box plots look similar and have lots of outliers

data_input_onehot_clipped = data_input_onehot.copy()
data_input_onehot_clipped.loc[:, 'Col1':'Col178'] = data_input_onehot_clipped.loc[:, 'Col1':'Col178'].clip(-600, 600)

# Check distplot
sns.distplot(data_input_onehot_clipped['Col165'])



from sklearn.preprocessing import StandardScaler

# Standard Scaling due to Gaussian-like Distribution
data_input_onehot_scaled = data_input_onehot_clipped.copy()
data_input_onehot_scaled.loc[:, :] = StandardScaler().fit_transform(data_input_onehot_scaled.loc[:, :])

data_input_onehot_scaled.head()


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 14))  
plt.title("Dendrogram")  

linkage_mat = shc.linkage(data_input_onehot_scaled, 
                          metric='cosine', 
                          method='average')

dend = shc.dendrogram(linkage_mat, 
                      p=5, 
                      truncate_mode='level')

plt.title("Dendrogram For Cosine Distance and Average Linkage", size=18, pad=25)
plt.xlabel("Nodes (Brackets Indicate Multiple Data Points)", size=15, labelpad=15)
plt.ylabel("Distance", size=15, labelpad=15)
plt.savefig('dend_avg.png', dpi=350)
from sklearn.cluster import AgglomerativeClustering

# Do Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=32, 
                                      affinity='cosine', 
                                      linkage='average')  

predicted_hier = agg_cluster.fit_predict(data_input_onehot_scaled) 

# Increment by 1 for confusion matrix 
predicted_hier = [i+1 for i in predicted_hier]

len(predicted_hier)
pd.Series(predicted_hier[:1300]).value_counts()

# Heatmap of confusion matrix

from sklearn.metrics import confusion_matrix 
conf_mat = confusion_matrix(class_labels.iloc[:1300].values, predicted_hier[:1300])[:5, :].T


# Plot
plt.figure(figsize=(10, 10))
ax= plt.subplot()
sns.heatmap(conf_mat, ax = ax, square=False, cmap='autumn_r', annot=True)

# Graph Formatting

ax.set_title('Confusion Matrix From True and Predicted Labels', size=13,
             pad=25)
ax.yaxis.set_ticklabels(list(range(1, 33)))
ax.xaxis.set_ticklabels([1, 2, 3, 4, 5])
ax.set_ylabel('Predicted Labels (from Hierarchical Clustering)', size=10,
              labelpad=15)
ax.set_xlabel('True Labels', size=10, labelpad=15)

ax

plt.savefig('2016A7TS0719G_fig10.png', dpi=350)

# Iterate over confusion matrix, get the true label for the predicted label

predicted_to_true_map = []
predicted_label=1

for predicted_label_row in conf_mat:
    
    detected_true_label = np.where(predicted_label_row == predicted_label_row.max())[0][0]+1
    predicted_to_true_map.append([predicted_label, detected_true_label])
    predicted_label+=1
    
predicted_to_true_map
# Mapping the predicted labels in the data to actual labels on the basis of calculated mapping

def auto_map_function(predicted_label_list, predicted_to_true_map):
    
    true_label_list = []
    for predicted_label in predicted_label_list:
        
        true_label = predicted_to_true_map[predicted_label-1][1]+100    
        true_label_list.append(true_label)
        
    true_label_list = [i-100 for i in true_label_list]
    
    return true_label_list
true_label_list = auto_map_function(predicted_hier, predicted_to_true_map)

# Check Rand-Score of mapping with known mapping
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(class_labels.iloc[:1300, 0].values, true_label_list[:1300])


# Create DataFrame for submission

submission_df = pd.DataFrame(true_label_list, columns=['Class'], index=data_input_onehot_scaled.index)
submission_df['id'] = submission_df.index
submission_df = submission_df[['id', 'Class']].reset_index(drop=True)
submission_df.head()
submission_df.shape

#Cropped 
predicted_11700 = submission_df.iloc[1300:, :]

from IPython.display import HTML 
import base64
import pandas as pd
import numpy as np

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    
    return HTML(html) 

create_download_link(predicted_11700)
