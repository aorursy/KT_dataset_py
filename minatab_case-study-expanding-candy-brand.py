import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df = pd.read_csv('/kaggle/input/fivethirtyeight-candy-power-ranking-dataset/candy-data.csv', index_col = 'competitorname')
#df.sort_values('winpercent',ascending=False).head()
df.dropna(inplace = True)
df.head()
taste_features = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy']
phys_features = ['nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus']
#ingredient correlation:
plt.figure(figsize = (20,8))        
sns.heatmap(df.loc[:,taste_features].corr(),annot=True, cmap = 'coolwarm')
#physical property correlation:
plt.figure(figsize = (20,8))        
sns.heatmap(df.loc[:,phys_features].corr(),annot=True, cmap = 'coolwarm')
#physical property correlation:
f, ax = plt.subplots(2,2, figsize = (16,10))
feature = "chocolate"
sns.heatmap(df.loc[df[feature]==1,['sugarpercent', 'pricepercent', 'winpercent']].corr(),annot=True, cmap = 'coolwarm', ax=ax[0,0])
sns.heatmap(df.loc[df[feature]==0,['sugarpercent', 'pricepercent', 'winpercent']].corr(),annot=True, cmap = 'coolwarm', ax=ax[0,1])    
ax[0,0].set_title(feature + " = 1")
ax[0,0].set_ylim([0,3])
ax[0,1].set_title(feature + " = 0")  
ax[0,1].set_ylim([0,3])

sns.regplot(x="sugarpercent", y="winpercent",
               truncate=True, data=df[df[feature]==1], ax = ax[1,0], label = "with "+feature, color = 'green')
sns.regplot(x="sugarpercent", y="winpercent",
               truncate=True, data=df[df[feature]==0], ax = ax[1,0], label = "without "+feature, color = 'red')
#ax[1,0].set_title(cat)

sns.regplot(x="pricepercent", y="winpercent",
               truncate=True, data=df[df[feature]==1], ax = ax[1,1], label = "with "+feature, color = 'green')
sns.regplot(x="pricepercent", y="winpercent",
               truncate=True, data=df[df[feature]==0], ax = ax[1,1], label = "without "+feature, color = 'red')
plt.legend()
plt.show()

#plt.tight_layout()    
#physical property correlation:
f, ax = plt.subplots(2,2, figsize = (16,10))
feature = "fruity"
sns.heatmap(df.loc[df[feature]==1,['sugarpercent', 'pricepercent', 'winpercent']].corr(),annot=True, cmap = 'coolwarm', ax=ax[0,0])
sns.heatmap(df.loc[df[feature]==0,['sugarpercent', 'pricepercent', 'winpercent']].corr(),annot=True, cmap = 'coolwarm', ax=ax[0,1])    
ax[0,0].set_title(feature + " = 1")
ax[0,0].set_ylim([0,3])
ax[0,1].set_title(feature + " = 0")  
ax[0,1].set_ylim([0,3])

sns.regplot(x="sugarpercent", y="winpercent",
               truncate=True, data=df[df[feature]==1], ax = ax[1,0], label = "with "+feature, color = 'green')
sns.regplot(x="sugarpercent", y="winpercent",
               truncate=True, data=df[df[feature]==0], ax = ax[1,0], label = "without "+feature, color = 'red')
#ax[1,0].set_title(cat)

sns.regplot(x="pricepercent", y="winpercent",
               truncate=True, data=df[df[feature]==1], ax = ax[1,1], label = "with "+feature, color = 'green')
sns.regplot(x="pricepercent", y="winpercent",
               truncate=True, data=df[df[feature]==0], ax = ax[1,1], label = "without "+feature, color = 'red')
plt.legend()
plt.show()

#plt.tight_layout()    
#physical property correlation:
f, ax = plt.subplots(2,2, figsize = (16,10))
feature = "peanutyalmondy"
sns.heatmap(df.loc[df[feature]==1,['sugarpercent', 'pricepercent', 'winpercent']].corr(),annot=True, cmap = 'coolwarm', ax=ax[0,0])
sns.heatmap(df.loc[df[feature]==0,['sugarpercent', 'pricepercent', 'winpercent']].corr(),annot=True, cmap = 'coolwarm', ax=ax[0,1])    
ax[0,0].set_title(feature + " = 1")
ax[0,0].set_ylim([0,3])
ax[0,1].set_title(feature + " = 0")  
ax[0,1].set_ylim([0,3])

sns.regplot(x="sugarpercent", y="winpercent",
               truncate=True, data=df[df[feature]==1], ax = ax[1,0], label = "with "+feature, color = 'green')
sns.regplot(x="sugarpercent", y="winpercent",
               truncate=True, data=df[df[feature]==0], ax = ax[1,0], label = "without "+feature, color = 'red')
#ax[1,0].set_title(cat)

sns.regplot(x="pricepercent", y="winpercent",
               truncate=True, data=df[df[feature]==1], ax = ax[1,1], label = "with "+feature, color = 'green')
sns.regplot(x="pricepercent", y="winpercent",
               truncate=True, data=df[df[feature]==0], ax = ax[1,1], label = "without "+feature, color = 'red')
plt.legend()
plt.show()

#plt.tight_layout()    
import numpy as np
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta
def makeCategorical(df,columns):
    for col in columns:
        df[col] = df[col].apply(lambda x:'with' if x==1 else "without")
makeCategorical(df,taste_features)
makeCategorical(df,phys_features)
df = df.dropna()
df.head()
corr_rat = {}
for f in taste_features+phys_features:
    for m in ['sugarpercent','pricepercent','winpercent']:
        corr_rat.setdefault(f,[]).append(correlation_ratio(df[f],df[m]))
corr_rat_df = pd.DataFrame.from_dict(data=corr_rat,orient='index', columns=['sugarpercent','pricepercent','winpercent'])
corr_rat_df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("How different features affect Sweetness, Price and Win probability?")
plt.show()
df['winpercent_bool'] = df['winpercent'].apply(lambda x:1 if x>75 else 0)
!pip install prince
import prince
famd = prince.FAMD(
     n_components=2,
     n_iter=10,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42)
famd = famd.fit(df.drop(['winpercent','winpercent_bool'], axis='columns'))
famd.explained_inertia_
import math
df_column_correlation = famd.column_correlations(df)
max_cor_len = 0
rot_ang = {}
for idx in df_column_correlation.index:
    point = df_column_correlation.loc[idx].values
    max_cor_len = max(math.sqrt(point[0]**2+point[1]**2),max_cor_len)
    rot_ang[idx] = math.atan(point[1]/point[0])
max_cor_len
circle1 = plt.Circle((0, 0), 1, color='k', fill=False)
circle2 = plt.Circle((0, 0), 1, color='k', fill=False)
circle3 = plt.Circle((0, 0), 1, color='k', fill=False)

f, ax = plt.subplots(1,3, figsize=(30,10))
ax[0].add_artist(circle1)  
ax[1].add_artist(circle2)  
ax[2].add_artist(circle3)  

max_len = 0
for idx in famd.column_correlations(df).index:
    idx_name = idx.split("_")
    point = (famd.column_correlations(df)).loc[idx].values
    point=point/max_cor_len    
    if idx_name[0] in taste_features and idx_name[1]=="with":
        ax[0].arrow(0, 0, point[0], point[1], head_width=0.05, head_length=0.05, color = 'k')
        #ax[0].annotate(idx_name[0],xy=tuple(point),xycoords='data', rotation=0, fontsize = 15, color='blue')
    elif idx_name[0] in phys_features and idx_name[1]=="with":
        ax[1].arrow(0, 0, point[0], point[1], head_width=0.05, head_length=0.05, color = 'k')
        #ax[1].annotate(idx_name[0],xy=tuple(point),xycoords='data', rotation=0, fontsize = 15, color='blue') 
    elif idx in ['sugarpercent','pricepercent']:
        ax[2].arrow(0, 0, point[0], point[1], head_width=0.05, head_length=0.05, color = 'k')
        #ax[2].annotate(idx_name[0],xy=tuple(point),xycoords='data', rotation=0, fontsize = 15, color='blue')        
for i,axis in enumerate(ax):
    axis.set_xlim([-1,1])
    axis.set_ylim([-1,1])
    axis.axis('equal')
    axis.set_xlabel('component 0')
    if i==0:
        axis.set_ylabel('component 1')
plt.show()
ax = famd.plot_row_coordinates(
     df,
     ax=None,
     figsize=(15, 15),
     x_component=0,
     y_component=1,
     labels=df.index,
     color_labels=['win {}'.format(t) for t in df['winpercent_bool']],
     ellipse_outline=True,
     ellipse_fill=True,
     show_points=True
)
winners = famd.row_coordinates(df)[df.winpercent_bool==1]
loosers = famd.row_coordinates(df)[df.winpercent_bool==0]
import numpy as np
wm_0, wm_1, w_2xconf_0, w2xconf_1,angle = prince.plot.build_ellipse(winners[0].astype(np.float),winners[1].astype(np.float))
lm_0, lm_1, l_2xconf_0, l2xconf_1,angle = prince.plot.build_ellipse(loosers[0].astype(np.float),loosers[1].astype(np.float))
#winner specifications:
winner_centroid = famd.inverse_transform([wm_0,wm_1])
winner_centroid.index = df_column_correlation.index
winner_centroid.columns = ['winner']
#winner specifications:
looser_centroid = famd.inverse_transform([lm_0,lm_1])
looser_centroid.index = df_column_correlation.index
looser_centroid.columns = ['looser']
pd.concat([winner_centroid,looser_centroid], axis = 1)
df_winner_looser = pd.concat([winner_centroid,looser_centroid], axis = 1)
for idx in df_winner_looser.index:
    if df_winner_looser.loc[idx,'winner']>df_winner_looser.loc[idx,'looser']:
        df_winner_looser.loc[idx,'winner'] = 1
        df_winner_looser.loc[idx,'looser'] = 0
    else:
        df_winner_looser.loc[idx,'winner'] = 0
        df_winner_looser.loc[idx,'looser'] = 1        
df_winner_looser.head(6)
with_idx = [idx for idx in df_winner_looser.index if idx.endswith('_with')]
df_winner_looser = df_winner_looser.loc[with_idx]
final_idx = [idx.split("_")[0] for idx in with_idx]
df_winner_looser.index = final_idx
df_winner_looser