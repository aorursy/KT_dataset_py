from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file

# data.csv has 18207 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'data.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.info()
df1.head(5)
df1.tail(3)
df1[df1['Name'].str.contains('Son')]
import re



def parse_numeric(s):

    """    

    유로화 문자열 -> 수치형 환산

    값 예시 : €0 또는 €43K

    끝에 K가 있으면 곱하기 1,000

    끝에 M이 있으면 곱하기 1,000,000    

    """

    if s[-1]  == 'K':

        return float(re.findall("\d+", s)[0])*1000    

    elif s[-1]  == 'M':

        return float(re.findall("\d+", s)[0])*1000000

    else:

        return float(re.findall("\d+", s)[0])



wage = df1['Wage'].apply(lambda x: parse_numeric(x))

#parse_numeric('123abc')
print(wage.describe())

np.log10(wage+0.1).plot(kind='hist') # 0이 있으므로 0.1을 더하자
df1.iloc[125][70:]  # 손흥민 stats
var_list = ['ID', 'Name', 'Value', 'Wage', 'Position', 'Age', 'Overall', 'Potential',

            'International Reputation', 'Weak Foot', 'Skill Moves', 'Crossing', 'Finishing', 

            'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 

            'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 

            'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 

            'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 

            'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']
# 잠시 팀별 wage나 value 순위가 궁금하다

df1_team = df1[var_list + ['Nationality', 'Club']]



# 데이터 파싱

for var in ['Value', 'Wage']:

    df1_team[var] = df1_team[var].apply(lambda x: parse_numeric(x))
def team_ranking(var, fn='sum', N=10):

    grouped = df1_team.groupby(['Club'])[var]

    if fn=='sum':

        return grouped.sum().sort_values(ascending=False)[:N]

    elif fn=='mean':

        return grouped.mean().sort_values(ascending=False)[:N]
team_ranking('Wage', fn='sum')
team_ranking('Age', fn='mean')
team_ranking('Value', fn='mean')
team_ranking('Overall', fn='mean')
team_ranking('Potential', fn='mean')
team_ranking('International Reputation', fn='mean')  # 뭐지????
df1_team.query(" Club == 'FC Barcelona' ")['International Reputation'].describe()
team_ranking('Composure', fn='mean')  # 뭐지????
df1_team.query(" Club == 'FC Barcelona' ")['Composure'].describe()
team_ranking('BallControl', fn='mean', N=20) # 뭐지????
df1_team.query(" Club == 'FC Barcelona' ")['BallControl'].describe()
team_ranking('ShortPassing', fn='mean')  # 뭐지????
df1_team.query(" Club == 'FC Barcelona' ")['ShortPassing'].describe()
df1_sub = df1[var_list]



# 골키퍼 제외, 골키퍼 스킬도 제외

df1_sub = df1_sub.query(" Position != 'GK' ")



# 데이터 파싱

for var in ['Value', 'Wage']:

    df1_sub[var] = df1_sub[var].apply(lambda x: parse_numeric(x))
print("Missing : ".format(df1_sub.isnull().sum(axis=0)))

df1_sub.describe().applymap('{:,.1f}'.format)
# 주급과의 상관계수

df1_sub.iloc[:, 2:].corr().iloc[1].sort_values(ascending=False)
plotScatterMatrix(df1_sub.iloc[:, 2:], 20, 10) 
#모델링 : Lasso

from sklearn.preprocessing import StandardScaler



X = df1_sub.iloc[:, 5:].values

scaler = StandardScaler()

X_std = scaler.fit_transform(X)
y = np.log( df1_sub['Wage'] + 0.1)
from sklearn import linear_model  # https://scikit-learn.org/stable/modules/linear_model.html



lasso = linear_model.Lasso(alpha=0.1)

lasso.fit(X_std, y)
df1_sub['pred_log'] = lasso.predict(X_std)

df1_sub['pred_euro'] = df1_sub['pred_log'].apply(lambda x: np.exp(x))
df1_sub[df1_sub['Name'].str.contains('Son')]
df1_sub[['Name', 'Wage','pred_euro']].iloc[:10]
import seaborn as sns

sns.scatterplot(x='Wage',y='pred_euro', data=df1_sub)
sns.lmplot(x='Wage', y='pred_euro', data=df1_sub)
sub = df1_sub.query("Wage > 10000")

sns.lmplot(x='Wage', y='pred_euro', data=sub)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 11)
plotScatterMatrix(df1, 20, 10)