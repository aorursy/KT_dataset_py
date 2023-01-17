import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import seaborn as sns

plt.style.use('fivethirtyeight')



path = "../input/"

os.chdir(path)

filenames = os.listdir(path)

df = pd.DataFrame()

for filename in sorted(filenames):

    try:

        read_filename = '../input/' + filename

        temp = pd.read_csv(read_filename,encoding='utf8')

        frame = [df,temp]

        df = pd.concat(frame)

    except UnicodeDecodeError:

        pass

    

df['Year'] = df.tourney_date.apply(lambda x: str(x)[0:4])

df['Sets'] = df.score.apply(lambda x: x.count('-'))

df['Rank_Diff'] =  df['loser_rank'] - df['winner_rank']

df['Rank_Diff_Round'] = df.Rank_Diff.apply(lambda x: 10*round(np.true_divide(x,10)))

df['ind'] = range(len(df))

df = df.set_index('ind')



bins = np.arange(10,200,10)

diff_df = pd.DataFrame()

diff_df['bins'] = bins

diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[df.Rank_Diff_Round == x]),(len(df[df.Rank_Diff_Round == x]) +len(df[df.Rank_Diff_Round == -x]))))

diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Grass')]),(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Grass')]) +len(df[(df.Rank_Diff_Round == -x) & (df.surface == 'Grass')]))))

diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Clay')]),(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Clay')]) +len(df[(df.Rank_Diff_Round == -x) & (df.surface == 'Clay')]))))

diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Hard')]),(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Hard')]) +len(df[(df.Rank_Diff_Round == -x) & (df.surface == 'Hard')]))))



plt.bar(diff_df.bins,diff_df.Prob,width = 9)

plt.ylim([0.5,0.9])

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')

plt.title('How likely are upsets?')
plt.plot(diff_df.bins,diff_df.Grass_Prob,'g')

plt.plot(diff_df.bins,diff_df.Hard_Prob,'b')

plt.plot(diff_df.bins,diff_df.Clay_Prob,'r')

plt.legend(['Grass','Hard','Clay'], loc = 2, fontsize = 12)

plt.ylim([0.5,0.9])

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')

plt.title('Upsets on Different Surfaces')
big_tour_df = df[df.draw_size == 128]

bins = np.arange(10,200,10)

diff_df = pd.DataFrame()

diff_df['bins'] = bins

diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]),(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]) +len(big_tour_df[big_tour_df.Rank_Diff_Round == -x]))))

diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Grass')]))))

diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Clay')]))))

diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Hard')]))))



plt.bar(diff_df.bins,diff_df.Prob,width = 9, color = 'r')

plt.ylim([0.5,0.9])

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')





big_tour_df = df[df.draw_size == 32]

bins = np.arange(10,200,10)

diff_df = pd.DataFrame()

diff_df['bins'] = bins

diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]),(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]) +len(big_tour_df[big_tour_df.Rank_Diff_Round == -x]))))

diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Grass')]))))

diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Clay')]))))

diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Hard')]))))



plt.bar(diff_df.bins,diff_df.Prob,width = 7)

plt.ylim([0.5,0.9])

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')

plt.xlim(0,175)

plt.legend(['Big Tournaments', 'Small Tournaments'], loc = 2, fontsize = 12)

plt.title('Upsets are more likely in small Tournaments')
def who_won(score,set_num):

    try:

        sets = score.split()

        set_score = sets[set_num-1]

        w = set_score[0]

        l = set_score[2]

        if int(w)>int(l):

            return 1

        if int(w)<int(l):

            return 0

    except ValueError:

        return -1

    except IndexError:

        return -1

df['1st_set'] = df.score.apply(lambda x: who_won(x,1))

df['2nd_set'] = df.score.apply(lambda x: who_won(x,2))

df['3rd_set'] = df.score.apply(lambda x: who_won(x,3))
def winning_per_set(Rank_diff, df, set_num):

    positive_diff_w = len(df[(df.Rank_Diff_Round == Rank_diff) & (df[set_num] == 1)])

    positive_diff_l = len(df[(df.Rank_Diff_Round == Rank_diff) & (df[set_num] == 0)])

    

    negative_diff_w = len(df[(df.Rank_Diff_Round == -Rank_diff) & (df[set_num] == 1)])

    negative_diff_l = len(df[(df.Rank_Diff_Round == -Rank_diff) & (df[set_num] == 0)])

    

    w = positive_diff_w + negative_diff_l

    l = positive_diff_l + negative_diff_w

    return np.true_divide(w, l + w)

 

bins = np.arange(10,200,10)

diff_df = pd.DataFrame()

diff_df['bins'] = bins

diff_df['Prob_1'] = diff_df.bins.apply(lambda x: winning_per_set(x,df,'1st_set'))

diff_df['Prob_2'] = diff_df.bins.apply(lambda x: winning_per_set(x,df,'2nd_set'))

diff_df['Prob_3'] = diff_df.bins.apply(lambda x: winning_per_set(x,df,'3rd_set'))





plt.plot(diff_df.bins,diff_df.Prob_1)

plt.plot(diff_df.bins,diff_df.Prob_2)

plt.plot(diff_df.bins,diff_df.Prob_3)

plt.legend(['Set 1','Set 2','Set 3'], loc = 2, fontsize = 12)

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')

plt.title('Upsets are more likely in the last set')
last_set = df[df.Sets == 2]

bins = np.arange(10,200,10)

diff_df = pd.DataFrame()

diff_df['bins'] = bins

diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[last_set.Rank_Diff_Round == x]),(len(last_set[last_set.Rank_Diff_Round == x]) +len(last_set[last_set.Rank_Diff_Round == -x]))))

diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Grass')]))))

diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Clay')]))))

diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Hard')]))))



plt.bar(diff_df.bins,diff_df.Prob,width = 9)

plt.ylim([0.5,0.9])

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')



last_set = df[df.Sets == 3]

bins = np.arange(10,200,10)

diff_df = pd.DataFrame()

diff_df['bins'] = bins

diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[last_set.Rank_Diff_Round == x]),(len(last_set[last_set.Rank_Diff_Round == x]) +len(last_set[last_set.Rank_Diff_Round == -x]))))

diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Grass')]))))

diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Clay')]))))

diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Hard')]))))



plt.bar(diff_df.bins,diff_df.Prob,width = 7, color = 'r')

plt.ylim([0.5,0.9])

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')



plt.legend(['2 Sets','3 Sets'], loc = 2, fontsize = 12)
df['bp_saving_rate_l'] = np.true_divide(df.l_bpSaved,df.l_bpFaced)

df['bp_saving_rate_w'] = np.true_divide(df.w_bpSaved,df.w_bpFaced)



plt.plot(df.winner_age,df.bp_saving_rate_w + np.random.normal(0,0.02,len(df)),'o', alpha = 0.1)

plt.plot(df.loser_age,df.bp_saving_rate_l + np.random.normal(0,0.02,len(df)),'o', alpha = 0.1)



x1 = df.winner_age[(np.isnan(df.winner_age) == 0) & (np.isnan(df.bp_saving_rate_w) == 0)]

y1 = df.bp_saving_rate_w[(np.isnan(df.winner_age) == 0) & (np.isnan(df.bp_saving_rate_w) == 0)]



x2 = df.loser_age[(np.isnan(df.loser_age) == 0) & (np.isnan(df.bp_saving_rate_l) == 0)]

y2 = df.bp_saving_rate_l[(np.isnan(df.loser_age) == 0) & (np.isnan(df.bp_saving_rate_l) == 0)]

plt.ylim([0.1,0.9])

plt.xlabel('Age')

plt.ylabel('Breaking Points Saving Rate')

plt.legend(['Winners','Losers'], loc = 4)

plt.title('Saving breaking points and Age')



print('Correlation between age and saving rates, Winners :',np.corrcoef(x1,y1)[1][0])

print('Correlation between age and saving rates, Losers :', np.corrcoef(x2,y2)[1][0])
df['Age_Diff'] = df.winner_age - df.loser_age

sns.kdeplot(df.Age_Diff)

sns.kdeplot(df.Age_Diff[df.Sets == 3])

plt.xlim([-15,15])

plt.legend(['All Matches', ' 3rd Set'])

plt.xlabel('Age Difference')

plt.title('Does the age difference kick in in the last set?')
sns.kdeplot(df.Age_Diff)

sns.kdeplot(df.Age_Diff[df['round'] == 'F'])

plt.xlim([-15,15])

plt.xlabel('Age Difference')

plt.legend(['All Matches','Finals'])

plt.title('Age Difference in the Finals')