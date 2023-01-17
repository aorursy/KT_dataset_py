import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from scipy import stats
df_original = pd.read_csv('../input/covid19-in-usa/us_states_covid19_daily.csv')

df_original.head()
ax = sns.heatmap(df_original.isnull(),yticklabels=False,cbar=False)

ax.set(xlabel='columns', ylabel='rows (white if null)', title='Checking Dataset for Null Values')

plt.show()
df_cleaning = df_original.copy()



# Update column types.

df_cleaning['date'] = df_cleaning['date'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m%d'))



# Rename colums for easier use.

df_cleaning = df_cleaning.rename(columns={'total':'total_tests_inc_pending', 

                                          'totalTestResults':'total_tests', 

                                          'death':'deaths'})

# Drop columns that aren't needed.

df_cleaning = df_cleaning.drop(columns=['dateChecked',

                                        'hash',

                                        'fips', 

                                        'deathIncrease',

                                        'hospitalizedIncrease',

                                        'negativeIncrease',

                                        'positiveIncrease',

                                        'totalTestResultsIncrease'])



state_dict = {}

for state in df_cleaning.state.unique():

    # Process each state separately, mostly to do the forward filling for NaNs by state.

    state_df = df_cleaning[df_cleaning['state']==state].copy()

    state_df = state_df.sort_values(by='date', ascending=True)

    state_df = state_df.reset_index(drop=True)

    state_df.loc[0] = state_df.loc[0].fillna(0)

    state_df.index = state_df.index + 1

    state_df = state_df.fillna(method='ffill')

    state_dict[state] = state_df

    

# Rejoin all states to make one large dataframe.

df = pd.DataFrame()

for state_df in state_dict.values():

    df = pd.concat([df, state_df])

df= df.reset_index()



# Add additional feature columns.

df['death_rate'] = (df['deaths'] / df['positive'])

# NaN values occur here when there are 0 positive cases and thus no deaths, can fill with 0s.

df['death_rate'] = df['death_rate'].fillna(0) 
ax = sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

ax.set(xlabel='columns', ylabel='rows (white if null)')

plt.show()
df.head()


def format_plot(fig):

    """

    Format figures for standard appearance.

    """

    plt.xlabel('Time')

    plt.xticks(rotation=45)

    plt.legend(loc='upper left', fontsize='large')



def best_and_worst(df, rank_on, y_label, states_per_plot=5, more_is_worse=True):

    """

    Plots states_per_plot number of best and worst faring states, ranked on the "rank_on" column.

    Labels both y axes with the y_label string provided.

    the more_is_worse boolean indicates if a larger number makes a state worse or better off.

    """

    most_recent_day = df.date.max()

    if more_is_worse:

        ranked = df[df['date']==most_recent_day].sort_values(by=rank_on, ascending=False)

    else:

        ranked = df[df['date']==most_recent_day].sort_values(by=rank_on, ascending=True)

    worst = ranked.head(states_per_plot).state

    best = ranked.tail(states_per_plot).state



    fig=plt.figure(figsize=(20,5))

    plt.subplot(1, 2, 1)

    for state in worst:

        state_df = df[df['state']== state].copy()

        plt.plot(state_df.date, state_df[rank_on], label=state)

        plt.title('Worst States')

        plt.ylabel(y_label)

        format_plot(fig)

    plt.subplot(1, 2, 2)

    for state in best:

        state_df = df[df['state']== state].copy()

        plt.plot(state_df.date, state_df[rank_on], label=state)

        plt.title('Best States')

        plt.ylabel(y_label)

        format_plot(fig)

        

    plt.show()

    

best_and_worst(df=df, rank_on='positive', y_label='Number of Positive Cases')
best_and_worst(df=df, rank_on='deaths', y_label='Number of Deaths')
best_and_worst(df=df, rank_on='total_tests', y_label='Number of Completed Tests', more_is_worse=False)
best_and_worst(df=df, rank_on='death_rate', y_label='Death Rate')
def get_lin_exp_fits(state_df, col='positive'):

    # Calculate linear and exponential fits.

    linear_coeffs = stats.linregress(x=state_df.index, y=state_df[col])

    positive_values = state_df[col]>0

    exp_coeffs = stats.linregress(x=state_df.index[positive_values], y=np.log(state_df.loc[positive_values, col]))

    return linear_coeffs, exp_coeffs



def plot_lin_vs_exp(state_df, col='positive', y_label='Number of Positive Cases'):

    """ Calculate linear and exponential fits 

    """

    linear_coeffs, exp_coeffs = get_lin_exp_fits(state_df, col=col)

    

    # Plot the results.

    fig=plt.figure(figsize=(20,5))

    plt.subplot(1, 3, 1)

    plt.plot(state_df.date, state_df[col], label="data")

    plt.plot(state_df.date,  linear_coeffs[1] + state_df.index*linear_coeffs[0], label="linear prediction")

    plt.title('Linear Fit')

    plt.ylabel(y_label)

    format_plot(fig)

    plt.subplot(1, 3, 2)

    plt.plot(state_df.date, state_df[col], label="data")

    plt.plot(state_df.date,  np.exp(exp_coeffs[1])  * (np.exp(exp_coeffs[0])**state_df.index), label="exp prediction")

    plt.title('Exponential Fit')

    plt.ylabel(y_label)

    format_plot(fig)

    plt.subplot(1, 3, 3)

    plt.plot(state_df.date, np.log(state_df[col]), label="data")

    plt.plot(state_df.date,  exp_coeffs[1] + state_df.index*exp_coeffs[0], label="exp prediction")

    plt.title('Exponential Fit, Log Plot')

    plt.ylabel('Log ' + y_label)

    format_plot(fig)

plot_lin_vs_exp(state_df=state_dict["NY"])
plot_lin_vs_exp(state_df=state_dict["NY"], col='deaths', y_label='Number of Deaths')

print("New York State")
plot_lin_vs_exp(state_df=state_dict["WA"])

print("Washington State:")
linear, exponential = get_lin_exp_fits(state_dict["NY"])

print ("linear r-squared:", linear.rvalue**2)

print ("exponential r-squared:", exponential.rvalue**2)
all_states = df.state.unique()



agg_df = pd.DataFrame(columns=['state','linear_slope','linear_intercept','linear_r2','exp_base','exp_mult','exp_r2'])



for state in all_states:

    state_df = state_dict[state]

    # Only consider states with at least one positive case

    if state_df.positive.abs().sum() != 0:

        linear, exp = get_lin_exp_fits(state_dict[state])

        row = pd.DataFrame(data={'state':[state],

                                 'linear_slope':[linear.slope],

                                 'linear_intercept':[linear.intercept],

                                 'linear_r2':[linear.rvalue**2],

                                 'exp_base':[np.exp(exp.slope)],

                                 'exp_mult':[np.exp(exp.intercept)],

                                 'exp_r2':[exp.rvalue**2]})

        agg_df = agg_df.append(row, ignore_index=True)



agg_df = agg_df.sort_values(by='exp_base', ascending=False)

agg_df.head()

    
plt.plot(agg_df.linear_r2, agg_df.exp_r2,'o')

plt.ylabel("Exponential Fit R Squared")

plt.xlabel("Linear Fit R Squared ")

plt.title("Linear and Exponential R Squared Values, Per State")

plt.show()
fig=plt.figure(figsize=(20,5))

agg_df = agg_df.sort_values(by='linear_r2')

plt.bar(agg_df.state,agg_df.linear_r2)

plt.xlabel("State")

plt.ylabel("Linear Fit R Squred Value")

plt.title("States Ranked by Linear Fit R Squared")

plt.show()
plot_lin_vs_exp(state_df=state_dict["TX"])

print("Texas: Less linear, more exponential")
plot_lin_vs_exp(state_df=state_dict["VI"])

print("Virginia: More linear, less exponential")
fig=plt.figure(figsize=(20,5))

agg_df = agg_df.sort_values(by='exp_base')

plt.bar(agg_df.state,agg_df.exp_base)

plt.xlabel("State")

plt.ylabel("Exponential Fit Basee Value")

plt.title("States Ranked by Exponential Fit Base")

plt.show()
fig=plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.plot(agg_df.linear_r2, agg_df.exp_base,'o')

plt.ylabel("Exponential Fit Base")

plt.xlabel("Linear Fit R Squared ")

plt.title("Linear R Squared vs Exponential Base, Per State")

plt.subplot(1,2,2)

plt.plot(agg_df.exp_r2, agg_df.exp_base,'o')

plt.ylabel("Exponential Fit Base")

plt.xlabel("Exponential Fit R Squared ")

plt.title("Exponential R Squared vs Exponential Base, Per State")

plt.show()