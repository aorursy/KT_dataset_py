%reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans # Class to develop kmeans model
from sklearn.metrics import silhouette_score # base for clustering
from yellowbrick.cluster import SilhouetteVisualizer

# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Use white grid plot background from seaborn
sns.set(font_scale=0.5, style="ticks")
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6})
import warnings
import os
%matplotlib inline
plt.rcParams.update({'figure.max_open_warning': 0}) #just to suppress warning for max plots of 20
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
os.chdir("../input/nyse/")
os.listdir()            # List all files in the folder
# Display output not only of last command but all commands in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Set pandas options to display results
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
#Let's try to analyze the dataset based on what is availiable with us
df = pd.read_csv("fundamentals.csv")
df.info()
df.describe()
security_df = pd.read_csv("securities.csv")
security_df.info()
security_df.describe()
df.columns = df.columns.str.strip().str.lower().str.replace(' \ ', '_').str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace(':', '').str.replace('\'', '').str.replace('\,', '').str.replace('\.', '').str.replace('&', '_')
df.columns = df.columns.str.strip().str.replace('__', '_').str.replace('/', '_')
df.columns = df.columns.str.strip().str.replace('__', '_').str.replace('___', '_')

#Cleaning Columns names of Securities dataframe as well
security_df.columns = security_df.columns.str.strip().str.lower().str.replace(' \ ', '_').str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace(':', '').str.replace('\'', '').str.replace('\,', '').str.replace('\.', '').str.replace('&', '_')
#let's verify whether cleaning columns names done or not?
df.columns
security_df.columns
missing_values = df.isna().sum()/len(df)

print("Total number of tuples:", len(df))
print("Percentage of  missing values:", round(missing_values.mean()*100,1),"%")

data_not_avlbl = []
for i in np.arange(0, len(df.columns), 10):
    data_not_avlbl.append(str(i)+"%");
plt.figure(figsize=[10,30]);

plt.yticks(np.arange(len(df.columns)), missing_values.index.values);
plt.xticks(np.arange(0, 1.1, .1), data_not_avlbl);

plt.ylim(0,len(df.columns));

plt.barh(np.arange(len(df.columns)), missing_values,color="crimson");
df=df.dropna()
df.reset_index(drop=True, inplace=True)
df.columns[df.isnull().any()] # Verify whether all NaN's have gone or not
df.index[df.isnull().any(axis=1)] # Verify (columnwise) whether all NaN's have gone or not
df_filtered=df
df_filtered.head()
#Let's first identify Categorical Features in the dataset
cat_features = df_filtered.dtypes[df_filtered.dtypes == 'object'].index.values 
cat_features
df_filtered.period_ending.unique()
len(df_filtered['period_ending'].str.slice(0, 7).unique())
df_filtered['year'] = pd.DatetimeIndex(df['period_ending']).year
df_filtered['month'] = pd.DatetimeIndex(df['period_ending']).month
df_filtered['day'] = pd.DatetimeIndex(df['period_ending']).day
df_filtered['month_year'] = pd.to_datetime(df_filtered['period_ending']).dt.to_period('M')
df_filtered['day'].astype('int');
df_filtered['year'].astype('int');
df_filtered['month'].astype('int');
def month1(x):
    if 0 < int(x) <= 3:
        return "1"            # Quarter 1
    if 3 < int(x) <= 6:
        return "2"            # Quarter 2
    if 6 < int(x) <= 9:
        return "3"            # Quarter 3
    if 9 < int(x) <= 12:
        return "4"            # Quarter 4
df_filtered['quarter'] = df_filtered['month'].map(lambda x : month1(x)) ;  # Which quarter clicked

df_filtered['leverage'] = df_filtered['total_assets'].divide(df_filtered['total_assets']-df_filtered['total_liabilities'])
df_filtered['leverage'].replace([np.inf, -np.inf], 0)
df_filtered['debtToEquity']=df_filtered['total_liabilities'].divide(df_filtered['total_equity'])
df_filtered['debtToEquity'].replace([np.inf, -np.inf], 0)
df_filtered['EBITtoAssets']=df_filtered['earnings_before_interest_and_tax'].divide(df_filtered['total_assets'])
df_filtered['EBITtoAssets'].replace([np.inf, -np.inf], 0)
df_filtered['EBITtoEquity']=df_filtered['earnings_before_interest_and_tax'].divide(df_filtered['total_equity'])
df_filtered['EBITtoEquity'].replace([np.inf, -np.inf], 0)
df_filtered['NCFtoEquity']=df_filtered['net_cash_flow-operating'].divide(df_filtered['total_equity'])
df_filtered['NCFtoEquity'].replace([np.inf, -np.inf], 0)
df_filtered['InvestToRetainedEarnings']=df_filtered['investments'].divide(df_filtered['retained_earnings'])
df_filtered['InvestToRetainedEarnings'].replace([np.inf, -np.inf], 0)
ax = df_filtered['quarter'].value_counts().plot(kind='bar',figsize=(8,4),color="crimson");
ax.set_xticklabels(['Q1', 'Q2','Q3','Q4'], rotation=0, fontsize=15);
df_filtered=df_filtered.copy()
grouped_df=df_filtered.groupby(['ticker_symbol'], as_index=True).pipe(lambda group:group.mean()).reset_index()
grouped_df.head()
grouped_df.shape
fig, ax = plt.subplots(1, 5, figsize=(18,4))

current_ratio_val = grouped_df['current_ratio'].values
cash_ratio_val = grouped_df['cash_ratio'].values
quick_ratio_val = grouped_df['quick_ratio'].values
earnings_per_share_val = grouped_df['earnings_per_share'].values
leverage_val = grouped_df['leverage'].values

sns.distplot(current_ratio_val, ax=ax[0], color='r');
ax[0].set_title('Distribution of Current Ratio', fontsize=14);
ax[0].set_xlim([min(current_ratio_val), max(current_ratio_val)]);

sns.distplot(cash_ratio_val, ax=ax[1], color='g');
ax[1].set_title('Distribution of Cash Ratio', fontsize=14);
ax[1].set_xlim([min(cash_ratio_val), max(cash_ratio_val)]);

sns.distplot(quick_ratio_val, ax=ax[2], color='b');
ax[2].set_title('Distribution of Quick Ratio', fontsize=14);
ax[2].set_xlim([min(quick_ratio_val), max(quick_ratio_val)]);

sns.distplot(earnings_per_share_val, ax=ax[3], color='r');
ax[3].set_title('Distribution of EPS', fontsize=14);
ax[3].set_xlim([min(earnings_per_share_val), max(earnings_per_share_val)]);

sns.distplot(earnings_per_share_val, ax=ax[4], color='g');
ax[4].set_title('Distribution of Leverage Ratio', fontsize=14);
ax[4].set_xlim([min(leverage_val), max(leverage_val)]);
plt.show();
fig=px.scatter(grouped_df, y="earnings_per_share", x="ticker_symbol", color="ticker_symbol", 
           hover_name="ticker_symbol",size="unnamed_0", size_max=10,animation_frame=grouped_df.year.astype(int));
fig.update_traces(textposition='middle center',marker={'symbol':"circle-x"}, textfont={'color':'black','family':'Helvetica','size':17},mode="text+markers");

fig.layout.update( title_text="Ticker_Symbol Vs Earning Per Share",title_font_size=20, showlegend=True,  transition= {'duration':10000 });
fig.show();
fig = px.scatter_ternary(grouped_df, a="debtToEquity", b="EBITtoEquity", c="NCFtoEquity", color="ticker_symbol", size="unnamed_0", hover_name="ticker_symbol",
                   size_max=15, color_discrete_map = {"debtToEquity": "blue", "EBITtoEquity": "green", "NCFtoEquity":"red"} )
fig.layout.update( title_text="Ternary Plot for debtToEquity Vs. EBITtoEquity Vs.  NCFtoEquity ",title_font_size=20, showlegend=True);
fig.show()
fig = plt.figure(figsize = (10,8))
sns.barplot(x = 'quarter',
            y = 'gross_profit',
            hue = 'year',       
            estimator = np.mean,
            ci = 95,
            data =df_filtered)
ticker_list=['unnamed_0']
symbol_name=df_filtered.ticker_symbol.unique()
parameters=['total_assets','total_liabilities','total_equity','earnings_before_interest_and_tax','net_cash_flow-operating',
                'investments','retained_earnings']

data_join = df_filtered[ticker_list + parameters].dropna().to_numpy(copy=True)

#data_join
ticker_list_y_ = data_join[:,0:1]
#ticker_list_y_
y = ticker_list_y_[:,0]
#y
X = data_join[:,1:]
#X

# total_liabilities vs. total_equity
plt.figure();
for i in range(len(symbol_name)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,2], X[idx,3], alpha=0.6, label=symbol_name[i]);
#plt.legend();
plt.xlabel('total_liabilities');
plt.ylabel('total_equity');
plt.show();

#  earnings_before_interest_and_tax  vs.  total_assets
plt.figure();
for i in range(len(symbol_name)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,4], X[idx,1], alpha=0.6, label=symbol_name[i]);
#plt.legend();
plt.xlabel('earnings_before_interest_and_tax');
plt.ylabel('total_assets');
plt.show();

# Scale data using StandardScaler
scaler = StandardScaler();
scaler.fit(X);
X_ = scaler.transform(X);
# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split( X,               # np array without target
                                               y,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
# Examine the results
X_train.shape              # (974, 7)
X_test.shape               # (325, 7)
clf = KMeans(n_clusters = 2)
#Train the object over data
clf.fit(X_train)

# What are our clusters?
clf.cluster_centers_
clf.cluster_centers_.shape         # (2, 7)
clf.labels_                        # Cluster labels for every observation
clf.labels_.size                   # 974
clf.inertia_                       # Sum of squared distance to respective centriods, SSE
silhouette_score(X_train, clf.labels_)    # 0.8814037609859733
#Make prediction over our test data and check accuracy
y_pred = clf.predict(X_test)
#y_pred
# How good is prediction
np.sum(y_pred == y_test)/y_test.size

# Are clusters distiguisable?
# We plot 1st and 2nd columns of X. Each point is coloured as per the cluster to which it is assigned (y_pred)
dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred);
# Scree plot:
sse = []
for i,j in enumerate(range(10)):
    # How many clusters?
    n_clusters = i+1
    # Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    # Train the kmeans object over data
    clf1.fit(X_train)
    # Store the value of inertia in sse
    sse.append(clf1.inertia_ )

#  Plot the line now
sns.lineplot(range(1, 11), sse)
visualizer = SilhouetteVisualizer(clf, colors='yellowbrick');
visualizer.fit(X_train) ;       # Fit the data to the visualizer
visualizer.show()     ;         # Finalize and render the figure
# t-SNE visualization
tsne = TSNE(n_components=2);
Xtsne = tsne.fit_transform(X_);

plt.figure();
for i in range(len(symbol_name)):
    idx = np.where(y == i)[0];
    plt.scatter(Xtsne[idx,0], Xtsne[idx,1], alpha=0.6, label=symbol_name[i]);
    
plt.title('t-SNE visualization');
#plt.legend();
plt.show();

df_fun_sec = pd.merge(df_filtered, security_df, on='ticker_symbol', how='inner')
df_fun_sec.head()
#Find Unique GICS Sectors
df_fun_sec['gics_sector'].unique()
df_fun_sec.shape
#Distribution of Sectors %agewise

labels = df_fun_sec['gics_sector'].astype('category').cat.categories.tolist()
counts = df_fun_sec['gics_sector'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True); #autopct is to show the % on plot
ax1.axis('equal')
plt.show();
ax=sns.catplot(x="gics_sector", y="gross_profit", hue="gics_sector",
            col="year", aspect=.6,
            kind="swarm", data=df_fun_sec);
ax.set_xticklabels(rotation=65, horizontalalignment='right')
ax=sns.catplot(x="gics_sector", y="earnings_per_share", kind="swarm", data=df_fun_sec);
ax.set_xticklabels(rotation=65, horizontalalignment='right')