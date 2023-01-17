# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# Data visualization
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns
%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
sns.set(style="whitegrid")

# Algorithms
from sklearn.ensemble import RandomForestClassifier
test_df = pd.read_csv("../input/test.csv")
train_df = pd.read_csv("../input/train.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

# Create categorical features
for i in range(1,5):
    train_df.loc[train_df['Wilderness_Area' + str(i)] == 1, 'Wilderness_Area'] = i
    test_df.loc[test_df['Wilderness_Area' + str(i)] == 1, 'Wilderness_Area'] = i
    
# Create categorical features
for i in range(1,40):
    train_df.loc[train_df['Soil_Type' + str(i)] == 1, 'Soil_Type'] = i
    test_df.loc[test_df['Soil_Type' + str(i)] == 1, 'Soil_Type'] = i
    
train_df.Soil_Type.fillna(41, inplace=True)
test_df.Soil_Type.fillna(41, inplace=True)

train_df.head(5)
numerical_features = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

OHE_features = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
       'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

categorical_features = ['Wilderness_Area', 'Soil_Type']

cover_type = {1:'Spruce/Fir',2:'Lodgepole Pine',3:'Ponderosa Pine',4 : 'Cottonwood/Willow',5 : 'Aspen',6:'Douglas-fir',7:'Krummholz'}

wilderness_areas ={1:'Rawah',2:'Neota',3:'Comanche Peak',4:'Cache la Poudre'}

soil_types = {1: 'Cathedral',2: 'Vanet - Ratake',3: 'Haploborolis',4: 'Ratake',5: 'Vanet',6: 'Vanet - Wetmore',7: 'Gothic',8: 'Supervisor - Limber',9: 'Troutville family',10: 'Rock outcrop',
11: 'Rock land',12: 'Legault',13: 'Catamount',14: 'Pachic Argiborolis',15: 'unspecified',16: 'Cryaquolis - Cryoborolis',17: 'Gateview',18: 'Rogert',19: 'Typic Cryaquolis',20: 'Typic Cryaquepts',
21: 'Typic Cryaquolls',22: 'Leighcan extremely bouldery',23: 'Leighcan - Typic Cryaquolls',24: 'Leighcan extremely stony',25: 'Leighcan warm, extremely stony',26: 'Granile - Catamount',27: 'Leighcan, warm',
28: 'Leighcan',29: 'Como - Legault',30: 'Como',31: 'Leighcan - Catamount',32: 'Catamount',33: 'Leighcan - Catamount - Rock outcrop',34: 'Cryorthents',35: 'Cryumbrepts',36: 'Bross',37: 'Rock - Cryumbrepts - Cryorthents',
38: 'Leighcan - Moran',39: 'Moran Leighcan',40: 'Moran Rock',41: 'NaN'}
# Visualize train and test distribution
def draw_train_test_kde(feature_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(train_df[feature_name], color=sns.color_palette("coolwarm",5)[0], label='Train')
    sns.kdeplot(test_df[feature_name], color=sns.color_palette("coolwarm",5)[4], label='Test')
    ax.set_title('Comparison of the ' + feature_name + ' distribution', size=20);

# Visualize traind distibution by caregorical feature
def draw_kde_cat_feat(feature_name, cat_feature_name, cat_feature_real_name):
    cat_feat_n = train_df[cat_feature_name].nunique()
    palette = sns.color_palette("viridis",cat_feat_n)
    fig, ax = plt.subplots(nrows=cat_feat_n, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(0,cat_feat_n):
        sns.kdeplot(train_df[train_df[cat_feature_name] == i+1][feature_name].values, clip_on=False, shade=True, alpha=1, lw=1.5, color=palette[i], ax=ax[i])
        sns.kdeplot(train_df[train_df[cat_feature_name] == i+1][feature_name].values, clip_on=False, color="w", lw=2, ax=ax[i])
        ax[i].text(0.01, 0.8, cat_feature_real_name[i+1], fontweight="bold", color=palette[i], ha="left", va="center", transform=ax[i].transAxes)
    ax[0].set_title('The forest ' + cat_feature_name + ' distributions for ' + feature_name, fontsize=18);
    ax[cat_feat_n-1].tick_params(axis='x',labelsize=13)
    
# It is a bad code. Someday I'll fix it.
def draw_kde_Soil_type(index_for_soil_type, feature_name, cat_feature_name, cat_feature_real_name):
    cat_feat_n = len(index_for_soil_type)
    palette = sns.color_palette("viridis",cat_feat_n)
    fig, ax = plt.subplots(nrows=cat_feat_n, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(cat_feat_n):
        sns.kdeplot(train_df[train_df[cat_feature_name] == index_for_soil_type[i]][feature_name].values, clip_on=False, shade=True, alpha=1, lw=1.5, color=palette[i], ax=ax[i])
        sns.kdeplot(train_df[train_df[cat_feature_name] == index_for_soil_type[i]][feature_name].values, clip_on=False, color="w", lw=2, ax=ax[i])
        ax[i].text(0.01, 0.8, cat_feature_real_name[i+1], fontweight="bold", color=palette[i], ha="left", va="center", transform=ax[i].transAxes)
    ax[0].set_title('The forest ' + cat_feature_name + ' distributions for ' + feature_name, fontsize=18);
    ax[cat_feat_n-1].tick_params(axis='x',labelsize=13)
    
#  Visualize the embedding vectors
def draw_3d_plot(data_pd, title_name, categorical_feature='Cover_Type', name_dict = cover_type):
    cat_feat_n = data_pd.loc[:,categorical_feature].nunique()
    palette = sns.color_palette("viridis",cat_feat_n)
    data = []
    
    for i in range(cat_feat_n):
        temp_trace = go.Scatter3d(
            x=data_pd[data_pd.loc[:,categorical_feature] == i+1]['First'],
            y=data_pd[data_pd.loc[:,categorical_feature] == i+1]['Second'],
            z=data_pd[data_pd.loc[:,categorical_feature] == i+1]['Third'],
            mode='markers',
            name=name_dict[i+1],
            marker=dict(
                size=3,
                color='rgb'+str(palette[i])
            )
        )
        data.append(temp_trace)

    layout = dict(title=title_name, autosize=True, 
                  scene=dict(xaxis=dict(title='First Cmp.', titlefont=dict(family='Arial, sans-serif',size=10,color='grey')), 
                             yaxis=dict(title='Second Cmp.', titlefont=dict(family='Arial, sans-serif',size=10,color='grey')), 
                             zaxis=dict(title='Third Cmp.', titlefont=dict(family='Arial, sans-serif',size=10,color='grey'))));
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
for feature in numerical_features:
    draw_train_test_kde(feature)
for feature in numerical_features:
    draw_kde_cat_feat(feature, 'Cover_Type', cover_type)
for feature in numerical_features:
    draw_kde_cat_feat(feature, 'Wilderness_Area', wilderness_areas)
index_for_soil_type = train_df['Soil_Type'].value_counts().index[:7]

for feature in numerical_features:
    draw_kde_Soil_type(index_for_soil_type, feature, 'Soil_Type', soil_types)
from sklearn import manifold, decomposition, ensemble, discriminant_analysis, random_projection
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_df_scale = scaler.fit_transform(train_df.loc[:,numerical_features + categorical_features])
%%time
rp = random_projection.SparseRandomProjection(n_components=3, random_state=42)
X_projected = rp.fit_transform(train_df_scale)
X_projected = pd.DataFrame(X_projected, columns=['First', 'Second', 'Third'])
X_projected = pd.concat([X_projected, train_df.loc[:,['Cover_Type', 'Wilderness_Area']]],axis=1)
X_projected.to_csv("SparseRandomProjection.csv", index=False) 
draw_3d_plot(X_projected, 'SparseRandom Projection')
%%time
X_pca = decomposition.TruncatedSVD(n_components=3, random_state=42).fit_transform(train_df_scale)
X_pca = pd.DataFrame(X_pca, columns=['First', 'Second', 'Third'])
X_pca = pd.concat([X_pca, train_df.loc[:,['Cover_Type', 'Wilderness_Area']]],axis=1)
X_pca.to_csv("TruncatedSVD.csv", index=False) 
draw_3d_plot(X_pca, 'Computing PCA projection')
%%time
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=3).fit_transform(train_df_scale, train_df.Cover_Type.values)
X_lda = pd.DataFrame(X_lda, columns=['First', 'Second', 'Third'])
X_lda = pd.concat([X_lda, train_df.loc[:,['Cover_Type', 'Wilderness_Area']]],axis=1)
X_lda.to_csv("LinearDiscriminantAnalysis.csv", index=False) 
draw_3d_plot(X_lda, 'Computing Linear Discriminant Analysis projection')
%%time
X_iso = manifold.Isomap(n_components=3).fit_transform(train_df_scale)
X_iso = pd.DataFrame(X_iso, columns=['First', 'Second', 'Third'])
X_iso = pd.concat([X_iso, train_df.loc[:,['Cover_Type', 'Wilderness_Area']]],axis=1)
X_iso.to_csv("Isomap.csv", index=False) 
draw_3d_plot(X_iso, 'Computing Isomap embedding')
%%time
clf = manifold.LocallyLinearEmbedding(n_components=3, method='standard', random_state=42)
X_lle = clf.fit_transform(train_df_scale)
X_lle = pd.DataFrame(X_lle, columns=['First', 'Second', 'Third'])
X_lle = pd.concat([X_lle, train_df.loc[:,['Cover_Type', 'Wilderness_Area']]],axis=1)
X_lle.to_csv("LLE_embedding.csv", index=False) 
draw_3d_plot(X_lle, 'Computing LLE embedding')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_df[numerical_features + categorical_features + OHE_features].values, train_df.Cover_Type.values)
predictions = rf.predict(test_df[numerical_features + categorical_features + OHE_features].values)
sub = pd.DataFrame({"Id": test_df.iloc[:,0].values,"Cover_Type": predictions})
sub.to_csv("rf.csv", index=False) 
