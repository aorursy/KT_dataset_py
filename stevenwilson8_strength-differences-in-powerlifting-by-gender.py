import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('../input/powerlifting-database/openpowerlifting.csv', low_memory = False)
df.head()
df.info()
cleaned_columns_df = df[['Name','Best3SquatKg','Best3BenchKg','Best3DeadliftKg',
                         'TotalKg','Sex','Equipment','BodyweightKg','Tested']]

cleaned_columns_df
cleaned_columns_df['Equipment'].value_counts()
filt = cleaned_columns_df['Equipment'] == 'Single-ply'
single_ply_df = cleaned_columns_df[filt].drop('Equipment', axis=1).reset_index(drop = True)
single_ply_df.info()
single_ply_df['Tested'].fillna('Enhanced', inplace = True)
single_ply_df['Tested'].replace('Yes', 'Natural', inplace = True)
single_ply_df['Tested'].value_counts()
compete_once_df = single_ply_df.groupby(['Name','Sex',
                       'Tested'])[['Best3SquatKg','Best3BenchKg',
                       'Best3DeadliftKg','TotalKg','BodyweightKg']].mean().reset_index()
compete_once_df = compete_once_df.dropna()
compete_once_df
compete_once_df['WeightClass'] = compete_once_df['BodyweightKg'].apply(float)

def weight_class(x):
    for i in range(10, 140, 10):
        if(x < i):
            return f"{str(i-10).zfill(3)} - {i} kg"
    return "130+ kg"
    
compete_once_df['WeightClass'] = compete_once_df['BodyweightKg'].apply( lambda x: weight_class(x))
compete_once_df
pivot_df = compete_once_df.groupby(['WeightClass','Tested','Sex'])['TotalKg'].count().reset_index()
pivot_df.pivot_table(columns=['Tested', 'Sex'], index=['WeightClass'], values='TotalKg')
filt = compete_once_df['WeightClass'].isin(['020 - 30 kg','030 - 40 kg', '120 - 130 kg', '130+ kg'])
compete_once_df.drop(index = compete_once_df[filt].index, inplace = True)

pivot_df = compete_once_df.groupby(['WeightClass','Tested','Sex'])['TotalKg'].count().reset_index()
pivot_df.pivot_table(columns=['Tested', 'Sex'], index=['WeightClass'], values='TotalKg')
clean_df = compete_once_df[compete_once_df['Tested'] == 'Natural'].drop('Tested', axis=1)
enhanced_df = compete_once_df[compete_once_df['Tested'] == 'Enhanced'].drop('Tested', axis=1)

lred, dred, lblue, dblue = ["#fb9a99", "#e31a1c", "#a6cee3", "#1f78b4"]

clean_df.groupby('Sex')['TotalKg'].count()
Male_series = clean_df[clean_df['Sex'] == 'M']['TotalKg']
Female_series = clean_df[clean_df['Sex'] == 'F']['TotalKg']

plt.close('all')
plt.figure(figsize = (14, 6))
sns.set_context("notebook", font_scale = 1.1)

sns.distplot(Female_series, label ='Female', color = dred)
sns.distplot(Male_series, label ='Male', color = dblue)
plt.legend()

plt.title('Distribution of Totals by Gender')
plt.yticks([])
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
plt.xlim(50, 950)
plt.xlabel('Total (Kg)')
plt.ylabel('Percentage of Competitors')
plt.show()
female_mean = Female_series.mean()
male_mean = Male_series.mean()

Mean_male_v_females = stats.percentileofscore(Female_series, male_mean)
Mean_female_v_males = stats.percentileofscore(Male_series, female_mean)
print(f'Average Female Total: {round(female_mean, 1)} kg')
print(f'Average Male Total: {round(male_mean, 1)} kg')
print()
print(f'Average male > {round(Mean_male_v_females, 1)}% of females')
print(f'Average female > {round(Mean_female_v_males, 1)}% of males')
D = male_mean - female_mean

female_sd = Female_series.std()
male_sd = Male_series.std()
D_sd = (female_sd**2 + male_sd**2)**0.5

print(f'Difference: {round(D,1)} kg')
print()
print(f'Female Standard Deviation: {round(female_sd, 1)} kg')
print(f'Male Standard Deviation: {round(male_sd, 1)} kg')
print(f"Difference's Standard Deviation: {round(D_sd, 1)} kg")
Difference_distribution = np.random.normal(D, D_sd, 100000)
plt.figure(figsize = (12, 6))
sns.set_context("notebook", font_scale = 1.1)

ax = sns.kdeplot(Difference_distribution, color = dblue, shade = True)
line = ax.get_lines()[-1]
x, y = line.get_data()
mask = x < 0
x, y = x[mask], y[mask]
ax.fill_between(x, y1=y, alpha=0.5, facecolor= dred, label = 'Female Lifts More')

mask = x >= 0
x, y = x[mask], y[mask]
ax.fill_between(x, y1=y, alpha=0.5, facecolor= dblue, label = 'Male Lifts More')


plt.title('Random Male Total - Random Female Total, Distribution')
# plt.axvline(x = 0, ymax = 0.4, label = '0 kg', color = 'red')
plt.legend()
plt.xlabel('Total (Kg)')

plt.yticks([])
plt.xlim(-400, 800)
# plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])

plt.show()
z_score = D/D_sd
print(f"Z-score: {round(z_score, 2)}")
print(f"Probability: {round(stats.norm.cdf(z_score)*100,2)}%")
E_Male_series = enhanced_df[enhanced_df['Sex'] == 'M']['TotalKg']
E_Female_series = enhanced_df[enhanced_df['Sex'] == 'F']['TotalKg']

E_D = E_Male_series.mean() - E_Female_series.mean()

E_female_sd = E_Female_series.std()
E_male_sd = E_Male_series.std()
E_D_sd = (E_female_sd**2 + E_male_sd**2)**0.5

print(f'Enhanced Difference: {round(E_D,1)} kg')
print(f"Enhanced Difference's Standard Deviation: {round(E_D_sd, 1)} kg")
E_z_score = E_D/E_D_sd
print(f"Z-score: {round(E_z_score, 2)}")
print(f"Probability: {round(stats.norm.cdf(E_z_score)*100,2)}%")
plt.clf()
MvF_df = clean_df.groupby(['WeightClass', 'Sex']).TotalKg.mean().reset_index()
plt.figure(figsize = (14, 7))
sns.set_context("notebook", font_scale = 1.1)
sns.set_palette([lred, lblue])

plt.xticks(rotation =20)
plt.title('Male Vs Female - Strength to Weight Distribution')
sns.set_context("talk")

sns.barplot(data = MvF_df, x = 'WeightClass', y = 'TotalKg', hue = 'Sex')
plt.ylim(100, 575)
plt.show()
s_score_df = clean_df.drop(['Best3SquatKg','Best3BenchKg','Best3DeadliftKg'], axis = 1)
s_score_df['T/BW'] = s_score_df['TotalKg']/s_score_df['BodyweightKg']

wc_s_score_df = s_score_df.groupby(['WeightClass'])['T/BW'].mean().reset_index()
wc_s_score_df.columns = ['WeightClass', 'WC-T/BW']

s_score_df = s_score_df.merge(wc_s_score_df, how='left')
s_score_df['S_score'] = s_score_df['T/BW']/s_score_df['WC-T/BW']
s_score_df.drop(columns = ['T/BW','WC-T/BW'], inplace = True)
s_score_df
E_Male_series = s_score_df[s_score_df['Sex'] == 'M']['S_score']
E_Female_series = s_score_df[s_score_df['Sex'] == 'F']['S_score']

E_D = E_Male_series.mean() - E_Female_series.mean()

E_female_sd = E_Female_series.std()
E_male_sd = E_Male_series.std()
E_D_sd = (E_female_sd**2 + E_male_sd**2)**0.5

print(f'S_score Difference: {round(E_D,2)*100}%')
print(f"S_score Difference's SD: {round(E_D_sd, 2)*100}%")
E_z_score = E_D/E_D_sd
print(f"Z-score: {round(E_z_score, 2)}")
print(f"Probability: {round(stats.norm.cdf(E_z_score)*100,2)}%")
resp_vars = ['Best3SquatKg','Best3BenchKg','Best3DeadliftKg','TotalKg']
MG_averages_df = compete_once_df.groupby(['Tested','Sex'])[resp_vars].mean().reset_index()
MG_averages_df
print("Female average strength as a proportion of male's:")
print(f"Upper body(bench press): {round(55.2/112.8 *100, 1)}%")
print(f"Lower body(squat): {round(106.8/177.6 *100, 1)}%")
print(f"Back(deadlift): {round(115.1/187.1 *100, 1)}%")
MG_percents_df = MG_averages_df[['Tested','Sex']].copy()

for var in resp_vars:
    MG_percents_df[var] = round(MG_averages_df[var]/MG_averages_df['TotalKg']*100, 1)

MG_percents_df['Tested-Sex'] = MG_percents_df['Sex'] + ' - ' + MG_percents_df['Tested']
MG_percents_df.drop(['TotalKg', 'Sex', 'Tested'], axis = 1, inplace=True)
MG_percents_df.columns = ['Squat/legs', 'Bench/chest', 'Deadlift/back', 'Tested-Sex']
MG_percents_df
MG_percents_df = pd.melt(frame=MG_percents_df, id_vars = ['Tested-Sex'], value_vars = ['Squat/legs',
    'Bench/chest', 'Deadlift/back'], value_name ="% of Total", var_name = 'Lift')
MG_percents_df = MG_percents_df.sort_values(['Tested-Sex'])
MG_percents_df
plt.close('all')

def display_figures(ax,df):
    show=df['% of Total'].to_list()
    i=0
    for p in ax.patches:
        h=p.get_height()
        if (h>0):
            value= str(show[i])+"%"
            ax.text(p.get_x()+p.get_width()/2,h+1, value, ha='center')
            i=i+1
            
plt.figure(figsize = (14, 7))
sns.set_context("notebook", font_scale = 1.1)
sns.set_palette([lred, dred, lblue, dblue])

ax = sns.barplot(data = MG_percents_df, x = 'Lift', y = '% of Total', hue ='Tested-Sex')
plt.ylim(15, 47)
plt.title('Muscle Group Strength Comparison')
# plt.yticks([15,25,35,45])
display_figures(ax, MG_percents_df)

plt.show()