# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# User settings

import matplotlib as mpl

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)



from collections import defaultdict
data = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data_sample = data.sample(frac=0.1, replace=True, random_state=1)
data.head()
data.info()
# 連続変数の統計量

data.describe()
# カテゴリカル変数の水準、欠損値確認

categorical_cols = data.dtypes[data.dtypes == "object"].index

for col in categorical_cols:

    display(col, data[col].value_counts(ascending=True, dropna=False))
import category_encoders as ce

df_session = data



ordinal_cols = ["Over18", "OverTime"]

ordinal_mapping_cols = [

    {"col": "Attrition", "mapping": {"No": 0, "Yes": 1}},

    {"col": "BusinessTravel", "mapping": {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}},

]

nominal_cols = ["Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]



# オーディナルエンコーディング

oe = ce.OrdinalEncoder(cols=ordinal_cols, handle_unknown="error")

df_session = oe.fit_transform(df_session)



oem = ce.OrdinalEncoder(mapping=ordinal_mapping_cols, handle_unknown="error")

df_session = oem.fit_transform(df_session)



# ワンホットエンコーディング

ohe = ce.OneHotEncoder(cols=nominal_cols, drop_invariant=True, handle_unknown="error")

df_cols_transformed = ohe.fit_transform(df_session)



(df_cols_transformed.dtypes=="int64").any()
df_cols_transformed.head()
df_cols_transformed_corr = df_cols_transformed.corr()

df_cols_transformed_corr["JobInvolvement"]
# NaNになっているのは値が一つ（分散が0）の列

for col in df_cols_transformed_corr["JobInvolvement"][df_cols_transformed_corr["JobInvolvement"].isnull()].index:

    display(df_cols_transformed[col].value_counts())
sns.heatmap(df_cols_transformed_corr)
# 相関係数が0.5より高い列を抽出

indices = np.where(df_cols_transformed_corr > 0.5)

indices = [(df_cols_transformed_corr.index[x], df_cols_transformed_corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]

indices
"""

Columns: JobInvolvement(順序)

"""
df = df_cols_transformed.copy()



fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))



ax.bar(x=df["JobInvolvement"].unique(), height=df["JobInvolvement"].value_counts())



# 目盛りの設定

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(200))

ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))



display(df["JobInvolvement"].unique(), df["JobInvolvement"].value_counts(), )
"""

Columns: JobInvolvement（順序）-MonthlyIncome（比例）

Graph: box plot

http://python-graph-gallery.com/boxplot/

https://seaborn.pydata.org/generated/seaborn.boxplot.html

visual variables:

    JobInvolvement: position, length

    MonthlyIncome: position, color saturation(>color hue)



"""



df = df_cols_transformed.copy()



fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(20,15))



# 描画

my_pal = {JobInvolvement: "r" if JobInvolvement == 2 else "lightgray" for JobInvolvement in df.JobInvolvement.unique()}

for axa, axb, axc in ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)):

    sns.boxplot(x=df["JobInvolvement"], y=df["MonthlyIncome"], ax=axa)

    sns.boxplot(x=df["JobInvolvement"], y=df["MonthlyIncome"], palette="Blues", ax=axb)

    sns.boxplot(x=df["JobInvolvement"], y=df["MonthlyIncome"], palette=my_pal, ax=axc)



# 軸関係の描画設定



# 軸関係全部消す

for ax in (ax5, ax6, ):

    ax.axis("off")



# 特定の軸の軸ラベル／目盛り／目盛りラベルを消す

for ax in (ax11, ax12, ):

    # ax.axes.xaxis.set_visible(False)

    ax.axes.yaxis.set_visible(False)



# 特定の軸の軸ラベルを消す

for ax in ():

    # ax.axes.set_xlabel("")

    ax.axes.set_ylabel("")



# 特定の軸の目盛り／目盛りラベルを消す

for ax in (ax8, ax9, ):

    # ax.axes.xaxis.set_ticks([])

    ax.axes.yaxis.set_ticks([])



# 特定の軸の目盛りラベルを消す

for ax in ():

    # ax.axes.xaxis.set_ticklabels([])

    ax.axes.yaxis.set_ticklabels([])



# 特定の軸の特定の方向の目盛りや目盛りラベルを消す

for ax in ():

    ax.tick_params(

        labelbottom=False, labelleft=False, labelright=False, labeltop=False,

        bottom=False, left=False, right=False, top=False

    )
"""

Columns: JobInvolvement（順序）-Department（名義）/RelationshipSatisfaction（順序）/JobSatisfaction（順序）

"""



df = data.copy()

df_grouped_dict = {}

target_columns = ["Department", "RelationshipSatisfaction", "JobSatisfaction"]



for i, col in enumerate(target_columns):

    df_grouped_dict[col] = df[[col, "JobInvolvement"]].groupby(col).mean()

    df_grouped_dict[col].index = df_grouped_dict[col].index.astype('str')
"""

Graph: bar plot

https://python-graph-gallery.com/barplot/

visual variables:

    JobInvolvement: length

    others: position

"""



fig, axs = plt.subplots(1, 3, figsize=(20, 7.5))



for i, col in enumerate(target_columns):

    axs[i].axes.set_title(col)

    axs[i].bar(df_grouped_dict[col].index, df_grouped_dict[col].JobInvolvement)

    for j, _ in enumerate(df_grouped_dict[col].index):

        axs[i].text(j, df_grouped_dict[col].JobInvolvement.iloc[j], round(df_grouped_dict[col].JobInvolvement.iloc[j], 3),

                    horizontalalignment="center", verticalalignment="bottom", color="dimgray")

        

axs[0].axes.set_ylabel("JobInvolvement")
"""

Graph: lollipop plot

https://python-graph-gallery.com/lollipop-plot/

https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.plot

https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.hlines

https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.text

https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.stem.html#matplotlib.pyplot.stem

visual variables:

    JobInvolvement: position, length

    others: position

    

僅かな差である場合、bar plotよりlolipop plotの方が視認しやすい（モアレを回避できる）

"""



fig, axs = plt.subplots(1, 3, figsize=(20, 7.5))



for i, col in enumerate(target_columns):

    axs[i].axes.set_title(col)

    axs[i].stem(df_grouped_dict[col].index, df_grouped_dict[col].JobInvolvement,

                basefmt="lightgray", markerfmt="o", use_line_collection=True)

    for j, _ in enumerate(df_grouped_dict[col].index):

        axs[i].text(j, 0, round(df_grouped_dict[col].JobInvolvement.iloc[j], 3),

                    horizontalalignment="center", verticalalignment="top", color="dimgray")

        

axs[0].axes.set_ylabel("JobInvolvement")
"""

Graph: horizontal lollipop plot

https://python-graph-gallery.com/lollipop-plot/

https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.plot

https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.hlines

https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.text

visual variables:

    JobInvolvement: position, length

    others: position

    

横にするとラベルが見やすくなる

"""



fig, axs = plt.subplots(1, 3, figsize=(20,5))

# fig.suptitle("JobInvolvement", fontsize=16)



for i, col in enumerate(target_columns):

    axs[i].axes.set_title(col)

    axs[i].hlines(y=df_grouped_dict[col].index, xmin=0, xmax=df_grouped_dict[col].JobInvolvement, color='skyblue')

    axs[i].plot(df_grouped_dict[col].JobInvolvement, df_grouped_dict[col].index, "o", color="dodgerblue")

    for j, _ in enumerate(df_grouped_dict[col].index):

        axs[i].text(df_grouped_dict[col].JobInvolvement.iloc[j]*9/10, j, round(df_grouped_dict[col].JobInvolvement.iloc[j], 3),

                    horizontalalignment="center", color="dimgray")



axs[1].axes.set_xlabel("JobInvolvement")
"""

bar plot/lolipop plot

ordered version

visual variables:

    JobInvolvement: position, length

    Department: position, color hue

    RelationshipSatisfaction, JobSatisfaction: position, color saturation

    

lolipop plotは遠くの変数との差を比較しにくいので、数値変数に従って並び替えをしたほうがよい

ただし、数値変数に従って並び替えをしないほうが良い場合（カテゴリ変数が順序変数の場合など）、bar plotを使用したほうがよい

また、以下では、数値変数に従って並び替えたあと、色の彩度によってこの順序情報を補うことを試してみる

しかし、lolipop plotは色付けしても見にくく（インクが少ないから）、matplotlibでも技術的には可能だが意味がない、という理由で色の変更オプションは実装されていない

"""



df_grouped_dict_ordered = {}

for col in target_columns:

    df_grouped_dict_ordered[col] = df_grouped_dict[col].sort_values(by='JobInvolvement', ascending=False)
fig, axs = plt.subplots(3, 3, figsize=(20, 18))



for i, col in enumerate(target_columns):

    axs[0][i].axes.set_title(col)

    axs[0][i].stem(df_grouped_dict_ordered[col].index, df_grouped_dict_ordered[col].JobInvolvement, basefmt="lightgray", use_line_collection=True)

    for j, _ in enumerate(df_grouped_dict_ordered[col].index):

        axs[0][i].text(j, df_grouped_dict_ordered[col].JobInvolvement.iloc[j]*9/10, round(df_grouped_dict_ordered[col].JobInvolvement.iloc[j], 3),

                    horizontalalignment="center", verticalalignment="bottom", color="dimgray")



colors_dict = {

    "Department": ["tab:blue", "tab:orange", "tab:green"],

    "RelationshipSatisfaction": ["#003366f0", "#003366b0", "#00336630", "#00336670"],

    "JobSatisfaction": ["#00336630", "#00336670", "#003366b0", "#003366f0"],

}



for i, col in enumerate(target_columns):

    # snsは元のデータの順序を無視して並び替えを自動で行うので、データの順序を変えておいても描画時にラベルの文字コード順に並び替えられるっぽい

    # ので、今回はmatplotlibのカラーパレットにてhueとsaturationを設定

    # palette = "Blues_d" if col in ("RelationshipSatisfaction", "JobSatisfaction") else None

    # sns.barplot(x=df_grouped_dict_ordered[col].index, y=df_grouped_dict_ordered[col].JobInvolvement, palette=palette, ax=axs[i])

    axs[1][i].bar(df_grouped_dict_ordered[col].index, df_grouped_dict_ordered[col].JobInvolvement)

    axs[2][i].bar(df_grouped_dict_ordered[col].index, df_grouped_dict_ordered[col].JobInvolvement,

               color=colors_dict[col])

    for j, _ in enumerate(df_grouped_dict_ordered[col].index):

        for k in (1, 2):

            axs[k][i].text(j, df_grouped_dict_ordered[col].JobInvolvement.iloc[j], round(df_grouped_dict_ordered[col].JobInvolvement.iloc[j], 3),

                        horizontalalignment="center", verticalalignment="bottom", color="dimgray")

        

axs[1][0].axes.set_ylabel("JobInvolvement")
"""

Columns: JobInvolvement（順序）-YearsSinceLastPromotion（比例）

"""



df = data
"""

Violin Plot

visual variables:

    JobInvolvement: position, color hue

    YearsSinceLastPromotion: position, length

    

1つまたは複数のグループの数値変数の分布を視覚化できる

箱ひげ図に比べて、各数値点において密度が分かるのが利点



ただしこの密度は、カーネル密度推定による描画なので注意が必要

特に小さなデータセットの場合は、サンプル数が小さいためその分布は正しくない可能性がある

代わりに、実際のサンプルをプロットするジッター付きの箱ひげ図が有効

https://www.data-to-viz.com/caveat/boxplot.html#boxplotjitter



https://seaborn.pydata.org/generated/seaborn.violinplot.html

https://seaborn.pydata.org/generated/seaborn.boxplot.html

"""



fig, axs = plt.subplots(1, 3, figsize=(15, 8))



# https://qiita.com/nkay/items/d1eb91e33b9d6469ef51#63-%E8%BB%B8%E3%81%AE%E6%9C%80%E5%B0%8F%E5%80%A4%E6%9C%80%E5%A4%A7%E5%80%A4%E3%81%AE%E8%A8%AD%E5%AE%9A

for ax in axs:

    ax.set_ylim(-3, 18)



sns.violinplot( x=df["JobInvolvement"], y=df["YearsSinceLastPromotion"], ax=axs[0])

sns.boxplot(x=df["JobInvolvement"], y=df["YearsSinceLastPromotion"], ax=axs[1])

sns.boxplot(x=df["JobInvolvement"], y=df["YearsSinceLastPromotion"], ax=axs[2])

sns.swarmplot(x=df["JobInvolvement"], y=df["YearsSinceLastPromotion"], color="grey", ax=axs[2])
"""

columns: YearsSinceLastPromotion（比例）-TotalWorkingYears（比例）

scatter with marginal point

YearsSinceLastPromotion: position, length, satulation()

TotalWorkingYears: position, length



2d density

YearsSinceLastPromotion: position, length, color satulation

TotalWorkingYears: position, length, color satulation

"""
"""

scatter with marginal point

YearsSinceLastPromotion: position, length, satulation()

TotalWorkingYears: position, length



jointplotはseabornのfigureレベルの関数なので、matplotlibのaxに入れられない

matplotlibのオブジェクトのはなし

https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9

seabornの関数には2種類（figureレベル、axesレベル）のものはあるというはなし

https://qiita.com/skotaro/items/7fee4dd35c6d42e0ebae#seaborn%E3%81%AE%E4%BE%BF%E5%88%A9%E3%83%97%E3%83%AD%E3%83%83%E3%83%88%E6%A9%9F%E8%83%BD%E3%81%AF%E4%BD%95%E3%82%92%E3%81%97%E3%81%A6%E3%81%84%E3%82%8B%E3%81%AE%E3%81%8B



"""



sns.jointplot(x=df["YearsSinceLastPromotion"], y=df["TotalWorkingYears"], kind='scatter', s=15, linewidth=2)
"""

2d density

YearsSinceLastPromotion: position, length, color satulation

TotalWorkingYears: position, length, color satulation

"""



sns.jointplot(x=df["YearsSinceLastPromotion"], y=df["TotalWorkingYears"], kind='kde', color="skyblue")