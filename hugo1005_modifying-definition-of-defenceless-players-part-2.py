from IPython.display import HTML
style = """
<style>
    .header1 { font-family:'Arial';font-size:30px; color:Black; font-weight:800;}
    .header2 { 
        font-family:'Arial';
        font-size:18px; 
        color:Black; 
        font-weight:600;
        border-bottom: 1px solid; 
        margin-bottom: 8px;
        margin-top: 8px;
        width: 100%;
        
    }
    .header3 { font-family:'Arial';font-size:16px; color:Black; font-weight:600;}
    .para { font-family:'Arial';font-size:14px; color:Black;}
    .flex-columns {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }
    .flex-container {
         padding: 20px;
    }
    
    .flex-container-large {
         padding: 20px;
         max-width: 40%;
    }
    
    .flex-container-small {
         padding: 20px;
         max-width: 17.5%;
    }
    
    .list-items {
        margin: 10px;
    }
    
    .list-items li {
        color: #3692CC;
        font-weight: 500;
    }
</style>
"""
HTML(style)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
import itertools
import datetime

def round_time(dt=None, round_to=60):
    if dt == None: 
        dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds+round_to/2) // round_to * round_to
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

## Helper function from sklearn docs
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    ax0.imshow(cm, interpolation='nearest', cmap=cmap)
    ax0.set_title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def plot_2d_space(X, y,ax_):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        ax_.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m)
    ax_.legend(loc='upper right', labels=['No Concussion', 'Concussion'])
data_selected_raw = pd.read_csv('../input/nfl-selected-features/data_selected.csv') 
data_selected = data_selected_raw.drop(columns=['Season_Year','GameKey','PlayID','GSISID'])
features = data_selected[['YardLineDist','ScoreDifference','Role','Temperature','Velocity','Play_Duration','Concussed']]

# One hot encode categorical role
role_enc = pd.get_dummies(features['Role'])
features_w_enc_role = features.merge(right=role_enc, how='inner', left_index=True, right_index=True)
features_w_enc_role = features_w_enc_role.drop(columns=['Role'])
features_w_enc_role = features_w_enc_role.dropna()

# Smote Oversampling
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_sample(features_w_enc_role.drop(columns=['Concussed']),features_w_enc_role['Concussed'])
plt.figure(figsize=(12,5))
ax0 = plt.subplot2grid((1,2),(0,0))
ax1 = plt.subplot2grid((1,2),(0,1))
ax0.set_title('Imbalanced dataset projection (Fig 1)')
ax1.set_title('SMOTE rebalanced dataset projection (Fig 2)')

# Imbalanced original data set
X_pca, y_pca = features_w_enc_role.drop(columns=['Concussed']),features_w_enc_role['Concussed']
pca = PCA(n_components=2)
X_pca_fit = pca.fit_transform(X_pca)
plot_2d_space(X_pca_fit, y_pca, ax0)

# Rebalance with SMOTE
X_pca, y_pca = X_sm, y_sm
pca = PCA(n_components=2)
X_sm_pca = pca.fit_transform(X_sm)
plot_2d_space(X_sm_pca, y_pca,ax1)
X_train, X_ct, y_train, y_ct = train_test_split(X_sm, y_sm, test_size=0.6, random_state=12)
# Cross validation set for model selection
X_cv, X_test, y_cv, y_test = train_test_split(X_ct, y_ct, test_size=0.75, random_state=12)
f1_scores = {}
confusion_matrices = {}

# This may take some time on your kernel be patient :)
for c_try in [0.001,0.01,0.1,1,10]:
    print("Using: %.4f" % c_try)
    clf_opt = SVC(probability=True, C=c_try)
    clf_opt.fit(X_train, y_train)
    
    y_predicted = clf_opt.predict(X_cv)

    model_f1_score = f1_score(y_cv, y_predicted)
    conf_matrix = confusion_matrix(y_cv, y_predicted)

    f1_scores[c_try] = [model_f1_score]
    confusion_matrices[c_try] = [conf_matrix]
print("F1 Score C=1: %.4f" % f1_scores[1][0])
print("F1 Score C=10: %.4f" % f1_scores[10][0])

plt.figure(figsize=(15,4))
ax1 = plt.subplot2grid((1,3), (0,0))
ax2 = plt.subplot2grid((1,3), (0,1))
ax3 = plt.subplot2grid((1,3), (0,2))

c_performance = pd.DataFrame(f1_scores).transpose()
c_performance_log = c_performance.copy()
c_performance_log.index = np.log10(c_performance_log.index)

c_performance_log.plot(ax=ax1)
ax1.set_xticks([-3,-2,-1,0,1])
ax1.set_xlabel('C Value (Log Base 10)')
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 score curve for varying C')

cf1 = np.array(confusion_matrices[1][0])
cf10 = np.array(confusion_matrices[10][0])

norm_cm = cf1.astype('float') / cf1.sum(axis=1)[:, np.newaxis]
sns.heatmap(norm_cm, ax=ax2, annot=True, fmt=".3f", cmap='Blues', yticklabels=['Normal', 'Concussion'], xticklabels=['Normal', 'Concussion'])
ax2.set_title('Confusion Matrix (C=1)')

norm_cm = cf10 / cf10.sum(axis=1)[:, np.newaxis]
sns.heatmap(norm_cm, ax=ax3, annot=True, fmt=".3f", cmap='Blues', yticklabels=['Normal', 'Concussion'], xticklabels=['Normal', 'Concussion'])
ax3.set_title('Confusion Matrix (C=10)')
# Undersample all data 185674 data points is too many for under 60 features, SVC is O(n^3) and is slow on large datasets
# Using SMOTE oversampled dataset C=1 (Optimal)
# Note random_state has been fixed to ensure when you build my code you get the same SVM as I do.

clf_init = SVC(probability=True, C=10)
clf_init.fit(X_train, y_train)
y_predicted = clf_init.predict(X_test)

model_f1_score = f1_score(y_test, y_predicted)
conf_matrix = confusion_matrix(y_test, y_predicted)
X_test_distances = pd.DataFrame(clf_init.decision_function(X_test))
X_train_distances = pd.DataFrame(clf_init.decision_function(X_train))
from sklearn.externals import joblib
joblib.dump(clf_init, 'nfl_concussions_model_w_probs_v2.pkl', compress=9)
print("F1 Score: %.4f" % model_f1_score)

plt.figure(figsize=(15,4))

ax1 = plt.subplot2grid((1,3), (0,0))
ax2 = plt.subplot2grid((1,3), (0,1))
ax3 = plt.subplot2grid((1,3), (0,2))

sns.distplot(X_train_distances[y_train == 1], color='r', kde=False, ax = ax1, norm_hist=True)
sns.distplot(X_train_distances[y_train == 0], color='b', kde=False, ax = ax1, norm_hist=True)

ax1.legend(['Concussion','No Concussion'])
ax1.set_xlim((-2,2))
ax1.axvline(x=1, linestyle='dashed', linewidth=0.8, c='black')
ax1.axvline(x=-1, linestyle='dashed', linewidth=0.8, c='black')
ax1.axvline(x=0, linestyle='dashed', linewidth=1, c='black')
ax1.set_title('Histogram of Projections \n Training Separation From Decision Boundary')

sns.distplot(X_test_distances[y_test == 1], color='r', kde=False, ax = ax2, norm_hist=True)
sns.distplot(X_test_distances[y_test == 0], color='b', kde=False, ax = ax2, norm_hist=True)

ax2.legend(['Concussion','No Concussion'])
ax2.set_xlim((-2,2))
ax2.axvline(x=1, linestyle='dashed', linewidth=0.8, c='black')
ax2.axvline(x=-1, linestyle='dashed', linewidth=0.8, c='black')
ax2.axvline(x=0, linestyle='dashed', linewidth=1, c='black')
ax2.set_title('Histogram of Projections \n Test Data Separation From Decision Boundary')

norm_cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(norm_cm, ax=ax3, annot=True, fmt=".3f", cmap='Blues', yticklabels=['Normal', 'Concussion'], xticklabels=['Normal', 'Concussion'])

ax1.set_ylim((0,1))
ax2.set_ylim((0,1))
ax3.set_title('Confusion Matrix')
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('Real Label')
# Helper Function
def platt_generate_graphs(cols,cols_non_binary,resolution=500, show_key=False, disable_plot=False):
    synthetic_probabilities = {}
    
    for col in cols:
        if col in cols_non_binary:
            x_min_col = x_min[col][0]
            x_max_col = x_max[col][0]
            x_range_col = x_max_col - x_min_col

            synthetic_range = np.arange(start=x_min_col,stop=x_max_col,step=x_range_col/500)
            X_synthetic = pd.concat([x_mean]*resolution, ignore_index=True)
        else:
            x_min_col = 0
            x_max_col = 1
            x_range_col = 1
            synthetic_range = np.arange(start=x_min_col,stop=x_max_col+1,step=x_range_col)
            X_synthetic = pd.concat([x_mean]*2, ignore_index=True)

        X_synthetic[col] = synthetic_range
        class_probabilities = clf_init.predict_proba(X_synthetic)

        synthetic_probabilities[col] = pd.DataFrame(index=synthetic_range,data=class_probabilities)
    
    if not disable_plot:   
        plt.figure(figsize=(20,190))
        PLOT_COLUMNS = 4

        mean_probability = clf.predict_proba(x_mean)[0]

        for (i, key) in enumerate(synthetic_probabilities):
            if key in cols_non_binary or show_key is False:
                ax1 = plt.subplot2grid((len(cols), PLOT_COLUMNS), (int(i/PLOT_COLUMNS),i%PLOT_COLUMNS))
                synthetic_probabilities[key][1].plot(ax=ax1, color='black', linewidth=1.2)
                ax1.set_title(key)
                ax1.axhline(y=mean_probability[1], linestyle='dashed', linewidth=0.8, color='red')

    return synthetic_probabilities

# Initialisation
X_stats = pd.DataFrame(X_sm).describe()
X_stats.columns = features_w_enc_role.drop(columns=['Concussed']).columns
X_concussed = features_w_enc_role[features_w_enc_role['Concussed'] == 1].drop(columns=['Concussed'])
cols = X_stats.columns

# Calculate sensitivity around each concussion example
concussed_synth_probs = []

for i in range(X_concussed.shape[0]):
    x_mean = pd.DataFrame(X_concussed.iloc[i,:]).transpose()
    x_max = pd.DataFrame(X_stats.loc['max',:]).transpose()
    x_min = pd.DataFrame(X_stats.loc['min',:]).transpose()

    synth_probs = platt_generate_graphs(
        X_stats.columns,['YardLineDist','ScoreDifference','Role','Temperature','Velocity','Play_Duration'], disable_plot=True)
    concussed_synth_probs.append(synth_probs)

# Compared to an average example (Maybe compare to avg noncussed be more meaningful)
x_mean = pd.DataFrame(X_stats.loc['mean',:]).transpose()
mean_probability = clf_init.predict_proba(x_mean)[0]
axes = []

# Sum over all sensitivity distributions
summed_synthetic_probabilities = concussed_synth_probs[0] 

for i in range(1,len(concussed_synth_probs)):
    for key in summed_synthetic_probabilities:
        summed_synthetic_probabilities[key] = summed_synthetic_probabilities[key] + concussed_synth_probs[i][key]

# Calculate an average agreement of feature sensitivity
avg_synthetic_probabilities = {}

for key in summed_synthetic_probabilities:     
    avg_synthetic_probabilities[key] = summed_synthetic_probabilities[key] / len(concussed_synth_probs) 
    
# Recombine all binary role features into single categorical
binary_role_features = [col for col in avg_synthetic_probabilities if col not in ['YardLineDist','ScoreDifference','Role','Temperature','Velocity','Play_Duration']]
role_recombined = {}

# We calculate the relative increase / decrease in probability of concussions as this is more visual
for key_nb in binary_role_features:
    role_recombined[key_nb] = [avg_synthetic_probabilities[key_nb][1][1] - avg_synthetic_probabilities[key_nb][1][0]]
    
role_probabilities = pd.DataFrame(role_recombined).transpose().sort_values(by=0)

axes = {}

# Plot the average sensitivity
plt.figure(figsize=(20,20))
# plt.suptitle('SVM Model Sensitivity Analysis By Feature')
PLOT_COLUMNS = 5

for (i, key) in enumerate(avg_synthetic_probabilities):
    if key not in binary_role_features:
        if len(axes) < i + 1:
            ax1 = plt.subplot2grid((3, PLOT_COLUMNS), (int(i/(PLOT_COLUMNS-2)),2 + i%(PLOT_COLUMNS-2)))
            axes[key] = ax1
        else:
            ax1 = axes[key]

        avg_synthetic_probabilities[key][1].plot(ax=ax1, color='black', linewidth=1.2)
        ax1.set_title(key)
        ax1.axhline(y=mean_probability[1], linestyle='dashed', linewidth=0.8, color='red')
        ax1.minorticks_on()
    
axr = plt.subplot2grid((3, PLOT_COLUMNS), (0,0), colspan=2, rowspan=2)
axr.set_title('Increase in concussion probability by Role')
role_probabilities.plot(kind='barh', ax=axr)

critical_ranges = {
    'YardLineDist': 0.2,
    'ScoreDifference': 0.15,
    'Temperature': 0.15,
    'Velocity': 0.55,
    'Play_Duration': 0.15,
}

# Annotations 
for (i, key) in enumerate(avg_synthetic_probabilities):
    if key not in binary_role_features:
        ax_ = axes[key]
        critical_values = avg_synthetic_probabilities[key][1] > critical_ranges[key]
        x_values = avg_synthetic_probabilities[key][critical_values][1].index.tolist()
        y_values = avg_synthetic_probabilities[key][critical_values][1].tolist()
        ax_.fill_between(x=x_values,y1=y_values, color='r', alpha=0.3, interpolate=True)
# Apply critical region restrictions
X_analysis = data_selected_raw[data_selected_raw['Concussed'] == 1]
yard_line_restrict =(X_analysis['YardLineDist'] >= 19) & (X_analysis['YardLineDist'] <= 35)
score_difference_restrict = (X_analysis['ScoreDifference'] >= -14) & (X_analysis['ScoreDifference'] <= 15)
temperature_restrict = (X_analysis['Temperature'] <= 64) & (X_analysis['Temperature'] >= 33)
velocity_restrict = X_analysis['Velocity'] <= 2.1
duration_restrict = X_analysis['Play_Duration'] <= 41

restricted_concussions = X_analysis[yard_line_restrict & score_difference_restrict & temperature_restrict & velocity_restrict & duration_restrict]

# Video Replay Data
video_data_2016 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
video_data_2017 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-control.csv')

video_data_2016.columns = video_data_2017.columns
video_data = video_data_2016.append(video_data_2017)

# Player Number
player_punt_data = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
# Some players have held more than one number / position
player_punt_data_dedupl = player_punt_data.groupby(by='GSISID').agg(' '.join)
restricted_concussions_number = restricted_concussions.merge(right=player_punt_data_dedupl, on='GSISID', how='left')

# Video - Selected Data
video_data_restricted = video_data.merge(right=restricted_concussions_number, how='inner', left_on=['season','gamekey','playid'], right_on=['Season_Year','GameKey','PlayID'])

#Video Review
video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
video_review_restricted = video_review.merge(right=video_data_restricted, how='inner', on=['Season_Year','GameKey','PlayID','GSISID'])
video_review_restricted = video_review_restricted[[c for c in video_review_restricted.columns if c[-2:] != '_y']]
video_review_restricted.columns = video_review_restricted.columns.str.replace('_x', '')
video_review_restricted['Primary_Partner_GSISID'] = video_review_restricted['Primary_Partner_GSISID'].astype('int64')

# Merging Concussion Partner Data
video_review_restricted_partner = video_review_restricted.merge(right=player_punt_data_dedupl, left_on='Primary_Partner_GSISID', right_on='GSISID', how='left', suffixes=('','_partner'))
video_review_restricted_partner = video_review_restricted_partner.merge(right=data_selected_raw, left_on=['Season_Year','GameKey','PlayID','Primary_Partner_GSISID'], right_on=['Season_Year','GameKey','PlayID','GSISID'], how='left', suffixes=('','_partner'))
video_review_restricted_partner[['Season_Year','GameKey','PlayID','GSISID','Number','Number_partner','Primary_Impact_Type','YardLineDist','ScoreDifference','Temperature','Velocity','Velocity_partner','Player_Activity_Derived','Primary_Partner_GSISID','Preview Link','Play_Duration']]
video_review_restricted_partner
video_review_restricted_partner['YardLineDist_Concussion'] = [21,15, 41, 14]
video_review_restricted_partner['Play_Duration_Concussion'] = [13,2, 12, 12]
video_review_restricted_partner['Play_Duration'][0] = 14
video_review_restricted_partner['Play_Duration'][1] = 10
video_review_restricted_partner['Play_Duration'][2] = 13
video_review_restricted_partner['Play_Duration'][3] = 12
axes = {}

# Plot the average sensitivity
plt.figure(figsize=(20,16))
# plt.suptitle('SVM Model Sensitivity Analysis By Feature')
PLOT_COLUMNS = 7

rows = video_review_restricted_partner[['GSISID','PlayID','Season_Year','GameKey']]
VIDEOS_TO_REPLAY =  rows.shape[0]

for k in range(VIDEOS_TO_REPLAY):
    GSISID_v,PLAYID_v,SEASON_v,GAMEKEY_v = rows.iloc[k,:].tolist()
    
    GSISID = video_review_restricted_partner['GSISID'] == GSISID_v
    PLAYID = video_review_restricted_partner['PlayID'] == PLAYID_v
    SEASON = video_review_restricted_partner['Season_Year'] == SEASON_v
    GAMEKEY = video_review_restricted_partner['GameKey'] == GAMEKEY_v
    selected_play = video_review_restricted_partner[GSISID & PLAYID & SEASON & GAMEKEY]

    for (i, key) in enumerate(avg_synthetic_probabilities):
        if key not in binary_role_features:
            if '%d_%s' % (k,key) not in axes:
                ax1 = plt.subplot2grid((VIDEOS_TO_REPLAY*3, PLOT_COLUMNS), (k*2 + int(i/(PLOT_COLUMNS-2)),2 + i%(PLOT_COLUMNS-2)), rowspan=2)
                axes['%d_%s' % (k,key)] = ax1
            else:
                ax1 = axes['%d_%s' % (k,key)]

            avg_synthetic_probabilities[key][1].plot(ax=ax1, color='grey', linewidth='1.2')
            ax1.set_title(key)
    #       ax1.axhline(y=mean_probability[1], linestyle='dashed', linewidth='0.8', color='red')
            ax1.minorticks_on()
    
    critical_ranges = {
        'YardLineDist': 0.18,
        'ScoreDifference': 0.15,
        'Temperature': 0.15,
        'Velocity': 0.50,
        'Play_Duration': 0.122,
    }

    axr = plt.subplot2grid((VIDEOS_TO_REPLAY*3, PLOT_COLUMNS), (k*2,0), colspan=2, rowspan=2)
    

    axr.set_title('SEASON: %d GAME KEY: %d PLAY ID: %d\n PLAYER ID: %d Role:' %(SEASON_v,GAMEKEY_v,PLAYID_v, GSISID_v))

    filtered_roles = role_probabilities[role_probabilities[0] >= 0]
    colors = np.array(['k']*filtered_roles.shape[0])
    colors[filtered_roles.index == selected_play['Role'].values[0]] = 'c'
    filtered_roles.plot(kind='barh',colors=''.join(colors), ax=axr)

    # Annotations 
    for (i, key) in enumerate(avg_synthetic_probabilities):
        if key not in binary_role_features:
            ax_ = axes['%d_%s' % (k,key)]
            critical_values = avg_synthetic_probabilities[key][1] > critical_ranges[key]
            x_values = avg_synthetic_probabilities[key][critical_values][1].index.tolist()
            y_values = avg_synthetic_probabilities[key][critical_values][1].tolist()
            
            ax_.fill_between(x=x_values,y1=y_values, color='r', alpha=0.3, interpolate=True)
            arrow_value = selected_play[key]
            ax_.annotate('',xy=(arrow_value, 0), xytext=(arrow_value, np.max(y_values)+0.05), arrowprops=dict(headwidth=5.5,width=0.8,facecolor='black'))
            ax_.set_ylim((0,np.max(y_values)+0.05))

            if key == 'Velocity':
                arrow_value_partner = selected_play["%s_partner" % key]
                
                # Deals with NANS
                if arrow_value_partner.values[0] > -1:
                    ax_.annotate('',xy=(arrow_value_partner, 0), xytext=(arrow_value_partner, np.max(y_values)+0.05), arrowprops=dict(headwidth=5.5,width=0.8,color='red'))
            
            if key == 'YardLineDist':
                arrow_value_partner = selected_play["YardLineDist_Concussion"]
                
                # Deals with NANS
                if arrow_value_partner.values[0] > -1:
                    ax_.annotate('',xy=(arrow_value_partner, 0), xytext=(arrow_value_partner, np.max(y_values)+0.05), arrowprops=dict(headwidth=5.5,width=0.8,color='blue'))
            
            if key == 'Play_Duration':
                arrow_value_concussion = selected_play["%s_Concussion" % key]
                ax_.annotate('',xy=(arrow_value_concussion, 0), xytext=(arrow_value_concussion, np.max(y_values)+0.05), arrowprops=dict(headwidth=5.5,width=0.8,color='green'))
            
plt.tight_layout()
X_analysis['YLD_Satisfy'] = yard_line_restrict.replace(to_replace = {True: 1, False: 0})
X_analysis['Score_Satisfy'] = score_difference_restrict.replace(to_replace = {True: 1, False: 0})
X_analysis['Temp_Satisfy'] = temperature_restrict.replace(to_replace = {True: 1, False: 0})
X_analysis['Velocity_Satisfy'] = velocity_restrict.replace(to_replace = {True: 1, False: 0})
X_analysis['Duration_Satisfy'] = duration_restrict.replace(to_replace = {True: 1, False: 0})
X_analysis['Satisfy_Count'] = X_analysis['YLD_Satisfy'] + X_analysis['Score_Satisfy'] + X_analysis['Temp_Satisfy'] + X_analysis['Velocity_Satisfy'] + X_analysis['Duration_Satisfy'] 

ax = X_analysis.groupby(by=['Satisfy_Count']).count()['GSISID'].plot(kind='barh', title='Frequency of the number of critical regions satisfied by concussion examples')
ax.set_ylabel('Number of critical regions satsified')
ax.set_xlabel('Concussion Examples')

