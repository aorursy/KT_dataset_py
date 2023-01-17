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
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
import bokeh

print(bokeh.__version__)
output_notebook()
df_user = pd.read_csv("../input/Users.csv")
df_user_achiv = pd.read_csv("../input/UserAchievements.csv")

print("No. of User: ", df_user.shape[0], df_user.shape)
print("No. of Users with achievment: ", df_user_achiv.shape[0], df_user_achiv.shape)
print(df_user.columns.values)
print(df_user.head(5))
print(df_user_achiv.columns.values)
print(df_user_achiv.head(5))
print(df_user_achiv['AchievementType'].unique())
df_user_achiv = df_user_achiv[df_user_achiv["AchievementType"] == "Competitions"]
print(df_user_achiv.shape)
df_user_user_achiv = pd.merge(df_user, df_user_achiv, left_on=['Id'], right_on=['UserId'], how='inner')
print(df_user_user_achiv.shape)
df_user_user_achiv.isnull().sum()
df_user_user_achiv.head(3)
df_user_user_achiv2 = df_user_user_achiv.drop(["AchievementType", "CurrentRanking","HighestRanking"], axis=1)
df_user_user_achiv2.head(3)
print(df_user_user_achiv2.shape)
df_user_user_achiv2.isnull().sum()
df_user_user_achiv3 = df_user_user_achiv2[df_user_user_achiv2["TierAchievementDate"].notnull()]
print(df_user_user_achiv3.shape)
df_user_user_achiv3.head(5)
# Convert the column type to datetime
df_user_user_achiv3['RegisterDate'] = pd.to_datetime(df_user_user_achiv3['RegisterDate'])
df_user_user_achiv3['TierAchievementDate'] = pd.to_datetime(df_user_user_achiv3['TierAchievementDate'])
df_user_user_achiv3.head(3)
df_user_user_achiv3['Reg_achiv_diff'] = df_user_user_achiv3['TierAchievementDate'] - df_user_user_achiv3['RegisterDate']
df_user_user_achiv3['Reg_achiv_diff'] = df_user_user_achiv3['Reg_achiv_diff']/np.timedelta64(1,'Y')
df_user_user_achiv3.head(3)
# Create dataframe for performance tier distribution
df_ptier = df_user_user_achiv3['PerformanceTier'].value_counts()
df_ptier = df_ptier.reset_index()
df_ptier = df_ptier.rename(columns={'index': 'PerformanceTier', 'PerformanceTier': 'No_of_users'})
print(df_ptier)
# show the performance tier distribution
perf_tier = figure(plot_width=800, plot_height=400, x_axis_label='No of Users', y_axis_label='Performance Tier', title="Performance Tier vs No. of Users")
perf_tier.hbar(y=df_ptier["PerformanceTier"], height=0.5, left=0, right=df_ptier["No_of_users"], color="green")
show(perf_tier)
df_u2 = df_user_user_achiv3.groupby(['RegisterDate'])[['Id_x']].sum()
df_u2 = df_u2.reset_index()
df_u2 = df_u2.rename(columns={'Id_x': 'no_of_users'})
print(df_u2.head(5))
# Number of users registered since 2010
users_growth = figure(plot_width=800, plot_height=400, x_axis_type="datetime", x_axis_label="Date", y_axis_label="No. of Registered Users")
users_growth.circle(df_u2['RegisterDate'],df_u2['no_of_users'] , line_color="#2E8B57",fill_color=None, size=5, legend='No. of users')
show(users_growth)
# Number of users registered since 2010
perf_growth = figure(plot_width=800, plot_height=400, x_axis_type="datetime", x_axis_label="Date", y_axis_label="Performance Tier")
df_user_5 = df_user_user_achiv3[df_user_user_achiv3['PerformanceTier'] ==5]
df_user_4 = df_user_user_achiv3[df_user_user_achiv3['PerformanceTier'] ==4]
df_user_3 = df_user_user_achiv3[df_user_user_achiv3['PerformanceTier'] ==3]
df_user_2 = df_user_user_achiv3[df_user_user_achiv3['PerformanceTier'] ==2]
df_user_1 = df_user_user_achiv3[df_user_user_achiv3['PerformanceTier'] ==2]
df_user_0 = df_user_user_achiv3[df_user_user_achiv3['PerformanceTier'] ==0]

perf_growth.circle(df_user_5['RegisterDate'],df_user_5['PerformanceTier'], line_color="#2E8B57",fill_color=None, size=5, legend='Tier 5')
perf_growth.circle(df_user_4['RegisterDate'],df_user_4['PerformanceTier'], line_color="#FBA40A",fill_color=None, size=5, legend='Tier 4')
perf_growth.circle(df_user_3['RegisterDate'],df_user_3['PerformanceTier'], line_color="#932567",fill_color=None, size=5, legend='Tier 3')
show(perf_growth)

df_user_user_achiv3['reg_year'] = df_user_user_achiv3['RegisterDate'].map(lambda x: x.year)
df_user_user_achiv3.head(3)
df_user_user_achiv32 = df_user_user_achiv3[np.abs(df_user_user_achiv3.Reg_achiv_diff-df_user_user_achiv3.Reg_achiv_diff.mean()) <= (2*df_user_user_achiv3.Reg_achiv_diff.std())]
df_user_user_achiv32 = df_user_user_achiv3[np.abs(df_user_user_achiv3.Reg_achiv_diff-df_user_user_achiv3.Reg_achiv_diff.mean()) > (2*df_user_user_achiv3.Reg_achiv_diff.std())]
print("Before outlier removed: ", df_user_user_achiv3.shape)
print("After outlier removed: ", df_user_user_achiv32.shape)
df_user_user_achiv32.head(3)
df_years_took = df_user_user_achiv32.groupby(['reg_year', 'PerformanceTier'])[['Reg_achiv_diff']].mean()
df_years_took2 = df_years_took.reset_index()
df_years_took2.head(10)
from bokeh.layouts import gridplot
def make_plot(year):
    plt_bar = figure(plot_width=350, plot_height=250, x_range=(1,8), x_axis_label='No.of Years', y_axis_label='Tiers', title="Users reg. on " + str(year))
    df_y = df_years_took2[df_years_took2['reg_year'] == year]
    plt_bar.hbar(y=df_y["PerformanceTier"], height=0.5, left=0, right=df_y["Reg_achiv_diff"], color="#35B778")
    return plt_bar
charts = [make_plot(year) for year in df_years_took2['reg_year'].unique()]
grid_plots = []
grid_row = [None]*3
for i, chrt in enumerate(charts):
    if i%3 == 0 and i!=0:
        grid_plots.append(grid_row)
        grid_row = [None]*3
        grid_row[i%3] = chrt
    else:
        grid_row[i%3] = chrt
grid_plots.append(grid_row)
show(gridplot(grid_plots))
