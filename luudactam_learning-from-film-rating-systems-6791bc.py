# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input/data"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Basic libraries

import numpy as np

import pandas as pd

from scipy import stats



# File related

import zipfile

from subprocess import check_output



# Machine Learning

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge

import tensorflow as tf



# Plotting

import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



plt.style.use('fivethirtyeight')



plt.rcParams['axes.labelsize'] = 20

plt.rcParams['axes.titlesize'] = 20

plt.rcParams['xtick.labelsize'] = 18

plt.rcParams['ytick.labelsize'] = 18

plt.rcParams['legend.fontsize'] = 14
with zipfile.ZipFile('../input/data/fandango.zip','r') as z: z.extractall('.')

    

print(check_output(["ls", "fandango"]).decode("utf8"))
fandango = pd.read_csv('fandango/fandango_score_comparison.csv')

fandango.head()
list(fandango)
fandango.describe()

# %%% List of films alphabetically sorted %%%



films_sorted = sorted(fandango['FILM'])



for i in films_sorted:

    print (i)
# Display list of keys (column names)



fandango.keys()
# WATCH OUT for the following typo: 'Metacritic_user_nom'



# Rename key



fandango.rename(columns={'Metacritic_user_nom':'Metacritic_user_norm'}, inplace=True)



fandango.keys()
# Set index

fandango.set_index('FILM')



# Sort by index

fandango.sort_values(by='FILM', ascending=True, inplace=True)



# Reset numerical index

fandango.reset_index(drop=True, inplace=True)



fandango.head()
fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(8.,6.))

plt.subplots_adjust(hspace=0.2)



fandango['Fandango_Stars'].plot.hist(

                                alpha=0.5,

                                bins=5,

                                label='Fandango_Stars',

                                ax=axes[0]

                                )



fandango['IMDB_norm'].plot.hist(

                            alpha=0.5,

                            bins=10,

                            label='IMDB_norm',

                            ax=axes[0]

                            )



axes[0].legend(loc='upper left')

axes[0].set_xlabel('Stars')

axes[0].set_xlim([0.,5.])

axes[0].set_ylim([0.,60.])



fandango['RT_user_norm'].plot.hist(

                            alpha=0.5,

                            bins=10,

                            label='RT_user_norm',

                            ax=axes[1]

                            )



fandango['RT_norm'].plot.hist(

                        alpha=0.5,

                        bins=10,

                        label='RT_norm',

                        ax=axes[1]

                        )



axes[1].legend(loc='upper left')

axes[1].set_title(' ')



fandango['Metacritic_user_norm'].plot.hist(

                                    alpha=0.5,

                                    bins=10,

                                    label='Metacritic_user_norm',

                                    ax=axes[2]

                                    )



fandango['Metacritic_norm'].plot.hist(

                                alpha=0.5,

                                bins=10,

                                label='Metacritic_norm',

                                ax=axes[2]

                                )



axes[2].legend(loc='upper left')

axes[2].set_title(' ')



plt.show()

plt.close()
fig, axes = plt.subplots()



rankings_lst = ['Fandango_Stars',

                'RT_user_norm',

                'RT_norm',

                'IMDB_norm',

                'Metacritic_user_norm',

                'Metacritic_norm']



fandango[rankings_lst].boxplot(vert=False)



axes.set_xlabel('Stars')



plt.show()

plt.close()
fig, axes = plt.subplots()



fandango['Fandango_Stars'].plot.hist(

                                alpha=0.5,

                                bins=5,

                                label='Fandango_Stars',

                                ax=axes

                                )



fandango['Fandango_Ratingvalue'].plot.hist(

                                    alpha=0.5,

                                    bins=10,

                                    label='Fandango_Ratingvalue',

                                    ax=axes

                                    )



axes.legend(loc='upper left')

axes.set_xlabel('Stars')

axes.set_xlim([0.,5.])

axes.set_ylim([0.,60.])



plt.show()

plt.close()
fig, axes = plt.subplots()



fandango[['Fandango_Stars', 'Fandango_Ratingvalue']].boxplot(vert=False)



axes.set_xlabel('Stars')



plt.show()

plt.close()
fig, axes = plt.subplots()



only_rt_80 = fandango['RT_norm'] >= 4.

rankings_lst = ['Fandango_Stars',

                'RT_user_norm',

                'IMDB_norm',

                'Metacritic_user_norm',

                'Metacritic_norm']



with matplotlib.style.context('fivethirtyeight'):

    fandango[rankings_lst].boxplot(vert=False)



with matplotlib.style.context('ggplot'):

    fandango[only_rt_80][rankings_lst].boxplot(vert=False)



axes.set_xlabel('Stars')



plt.title('Red boxes: RT best movies only')



plt.show()

plt.close()
rankings_lst = ['Fandango_Stars',

                'RT_user_norm',

                'RT_norm',

                'IMDB_norm',

                'Metacritic_user_norm',

                'Metacritic_norm']



def plot_heatmap(df):

    

    import seaborn as sns

    

    fig, axes = plt.subplots()



    sns.heatmap(df, annot=True)



    plt.show()

    plt.close()

    

plot_heatmap(fandango[rankings_lst].corr(method='pearson'))
plot_heatmap(fandango[only_rt_80][rankings_lst].corr(method='pearson'))
fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(8.,6.))

plt.subplots_adjust(hspace=0.4)



axes[0].scatter(

            fandango['RT_norm'],

            fandango['RT_user_norm'],

            color='black',

            alpha=0.5

            )



axes[0].scatter(

            fandango[only_rt_80]['RT_norm'],

            fandango[only_rt_80]['RT_user_norm'],

            color='red',

            alpha=0.5

            )



axes[0].set_ylabel('Stars')

axes[0].set_xlim([0.,5.])

axes[0].set_ylim([0.,5.5])

axes[0].set_title('RT versus RT users')



axes[1].scatter(

            fandango['RT_norm'],

            fandango['Metacritic_norm'],

            color='black',

            alpha=0.5

            )



axes[1].scatter(

            fandango[only_rt_80]['RT_norm'],

            fandango[only_rt_80]['Metacritic_norm'],

            color='red',

            alpha=0.5

            )



axes[1].set_ylabel('Stars')

axes[1].set_title('RT versus Metacritic')



axes[2].scatter(

            fandango['RT_norm'],

            fandango['IMDB_norm'],

            color='black',

            alpha=0.5

            )



axes[2].scatter(

            fandango[only_rt_80]['RT_norm'],

            fandango[only_rt_80]['IMDB_norm'],

            color='red',

            alpha=0.5

            )



axes[2].set_ylabel('Stars')

axes[2].set_title('RT versus IMDB')



axes[3].scatter(

            fandango['Metacritic_norm'],

            fandango['Fandango_Stars'],

            color='black',

            alpha=0.5

            )



axes[3].scatter(

            fandango[only_rt_80]['Metacritic_norm'],

            fandango[only_rt_80]['Fandango_Stars'],

            color='red',

            alpha=0.5

            )



axes[3].set_ylabel('Stars')

axes[3].set_xlabel('Stars')

axes[3].set_title('Metacritic versus Fandango')



plt.show()

plt.close()
# create a feature matrix 'X' by selecting two DataFrame columns

feature_cols = ['RT_user_norm', 'RT_norm', 'Metacritic_user_norm', 'Metacritic_norm']

X = fandango.loc[:, feature_cols]



# create a response vector 'y' by selecting a Series

y = fandango['IMDB_norm']



# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=43)

# Change 'random_state' value to obtain different final results
# Train model

reg = LinearRegression()

reg.fit(X_train, y_train)
# Best-fit coefficients of the linear regression

reg.coef_
# 'intercept' coefficient, i.e. independent coefficient

reg.intercept_
# Use the fitted model to make predictions for the testing set

y_pred = reg.predict(X_test)
learnt_df = X_test.copy(deep=True)



learnt_df.insert(loc=0,

                 column='IMDB_norm_predicted',

                 value=pd.Series(data=y_pred, index=learnt_df.index)

                )



learnt_df.insert(loc=0,

                 column='IMDB_norm_actual',

                 value=y_test

                )



learnt_df[['IMDB_norm_actual', 'IMDB_norm_predicted']].head()
# CHECK if predicition column is consistent with best-fit parameters



test_pred = sum(reg.coef_ * learnt_df.loc[45, :].values[2:]) + reg.intercept_



print('Prediction (index=45): ' + str(test_pred))
fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(8.,6.))

plt.subplots_adjust(hspace=0.4)



dot1 = axes[0].scatter(

                fandango['Metacritic_norm'],

                fandango['IMDB_norm'],

                color='blue',

                alpha=0.5

                )



dot2 = axes[0].scatter(

                learnt_df['Metacritic_norm'],

                learnt_df['IMDB_norm_predicted'],

                color='red',

                alpha=0.5

                )



axes[0].set_ylabel('Stars')

axes[0].set_xlim([0.,5.])

axes[0].set_ylim([0.,5.5])

axes[0].set_title('Metacritic versus IMDB')

axes[0].legend((dot1, dot2),

           ('full dataset', 'predicted'),

           scatterpoints=1,

           loc='lower right',

           ncol=3

           )



axes[1].scatter(

            fandango['Metacritic_user_norm'],

            fandango['IMDB_norm'],

            color='blue',

            alpha=0.5

            )



axes[1].scatter(

            learnt_df['Metacritic_user_norm'],

            learnt_df['IMDB_norm_predicted'],

            color='red',

            alpha=0.5

            )



axes[1].set_ylabel('Stars')

axes[1].set_title('Metacritic users versus IMDB')



axes[2].scatter(

            fandango['RT_norm'],

            fandango['IMDB_norm'],

            color='blue',

            alpha=0.5

            )



axes[2].scatter(

            learnt_df['RT_norm'],

            learnt_df['IMDB_norm_predicted'],

            color='red',

            alpha=0.5

            )



axes[2].set_ylabel('Stars')

axes[2].set_title('RT versus IMDB')



axes[3].scatter(

            fandango['RT_user_norm'],

            fandango['IMDB_norm'],

            color='blue',

            alpha=0.5

            )



axes[3].scatter(

            learnt_df['RT_user_norm'],

            learnt_df['IMDB_norm_predicted'],

            color='red',

            alpha=0.5

            )



axes[3].set_ylabel('Stars')

axes[3].set_title('RT users versus IMDB')

axes[3].set_xlabel('Stars')



plt.show()

plt.close()
fig, axes = plt.subplots()



axes.scatter(y_test, y_pred, color='red', alpha=0.5)

axes.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=1)

axes.set_xlabel('Actual')

axes.set_ylabel('Predicted')



plt.show()

plt.close()
# %%% Measuring the quality of the regression %%%



# Minimum of chi-squared obtained in the regression

min_chi2 = ((y_test - y_pred)**2).values



# Number of degrees of freedom for 'len(feature_cols)' parameters

n_degrees = len(y_test) - len(feature_cols)



def func_p_value(c,n):

    """

    

    c : chi-squared value

    n : number of degree of freedom (d.o.f.), 

        i.e. number of points subtracted by number of parameters

    

    Notice: p=0 is considered the worst possible fit and p=1 is

    considered to be the perfect fit. For example,

    

    In[235]: print(func_p_value(0, 45))

    Out[235]: 1.0

    

    In[236]: print(func_p_value(100, 45))

    Out[236]: 4.67686463534e-06   

  

    """

    return (1. - stats.chi2.cdf(c, n))



# p-value

p_value = func_p_value(sum(min_chi2), n_degrees)
fig, axes = plt.subplots()



markerline, stemlines, baseline = plt.stem(y_test, min_chi2)

plt.setp(markerline, linewidth=1, color='red', alpha=0.5)

plt.setp(stemlines, linewidth=1, color='red', alpha=0.5)

plt.setp(baseline, linewidth=0, color='gray', alpha=0.5)



axes.set_xlabel('Stars')

axes.set_ylabel(r'$\chi^2$ ')

axes.set_ylim(ymin=-0.01)



axes.set_title(r'$\chi^2_{\mathrm{total}} =$' + str(sum(min_chi2)) +

               r' ,  d.o.f.$=$' + str(n_degrees) + '\n'

               r'$\chi^2_{\mathrm{total}}/\mathrm{d.o.f.} =$' +

               str(sum(min_chi2)/n_degrees) + '\n'

               r'p-value $=$' + str(p_value)

               )



plt.show()

plt.close()
n_dim = len(feature_cols)



# Include extra dimension for independent coefficient

n_dim += 1



P = tf.placeholder(tf.float32,[None,n_dim])

q = tf.placeholder(tf.float32,[None,1])

T = tf.Variable(tf.ones([n_dim,1]))



bias = tf.Variable(tf.constant(1.0, shape = [n_dim]))

q_ = tf.add(tf.matmul(P, T),bias)



cost = tf.reduce_mean(tf.square(q_ - q))



learning_rate = 0.01



training_step = tf.train.GradientDescentOptimizer(

                    learning_rate=learning_rate

                    ).minimize(cost)



# Include extra column 'independent' for independent coefficient

X_train = X_train.assign(

            independent = pd.Series([1] * len(y_train),

            index=X_train.index)

            )



X_test = X_test.assign(

            independent = pd.Series([1] * len(y_train),

            index=X_test.index)

            )



# Convert panda dataframes to numpy arrays

P_train = X_train.as_matrix(columns=None)

P_test = X_test.as_matrix(columns=None)



q_train = np.array(y_train.values).reshape(-1,1)

q_test = np.array(y_test.values).reshape(-1,1)
training_epochs = 1000



with tf.Session() as sess:



    tf.global_variables_initializer().run()



    cost_history = np.empty(shape=[1], dtype=float)

    t_history = np.empty(shape=[n_dim, 1], dtype=float)



    for epoch in range(training_epochs):

    

        sess.run(

            training_step,

            feed_dict={P: P_train, q: q_train}

                )

        

        cost_history = np.append(

            cost_history,

            sess.run(cost, feed_dict={P: P_train, q: q_train})

        )

    

        t_history = np.append(

            t_history,

            sess.run(T, feed_dict={P: P_train, q: q_train}),

            axis=1

        )

    

    q_pred = sess.run(q_, feed_dict={P: P_test})[:, 0]

    

    mse = tf.reduce_mean(tf.square(q_pred - q_test))

    

    sess.close()
fig, axes = plt.subplots()



plt.plot(range(len(cost_history)), cost_history)



axes.set_xlim(xmin=0.95)

axes.set_ylim(ymin=1.e-2)



axes.set_xscale("log", nonposx='clip')

axes.set_yscale("log", nonposy='clip')



axes.set_ylabel('Cost')

axes.set_xlabel(r'Iterations')



plt.show()

plt.close()
# Print again result obtained with scikit-learn



print(' --- scikit-learn ---')

print(learnt_df[['IMDB_norm_actual', 'IMDB_norm_predicted']].head())



# NOW, TensorFlow



del learnt_df



learnt_df = X_test.copy(deep=True)



learnt_df.insert(loc=0,

                 column='IMDB_norm_predicted',

                 value=pd.Series(data=q_pred, index=learnt_df.index)

                )



learnt_df.insert(loc=0,

                 column='IMDB_norm_actual',

                 value=q_test

                )



print('')

print(' --- TensorFlow ---')

print(learnt_df[['IMDB_norm_actual', 'IMDB_norm_predicted']].head())
fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10.,6.))



axes[0].scatter(y_test, y_pred, color='red', alpha=0.5)

axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=1)

axes[0].set_xlabel('Actual')

axes[0].set_ylabel('Predicted')

axes[0].set_title('scikit-learn')



axes[1].scatter(q_test, q_pred, color='blue', alpha=0.5)

axes[1].plot([q_test.min(), q_test.max()], [q_test.min(), q_test.max()], '--', lw=1)

axes[1].set_xlabel('Actual')

axes[1].set_ylabel('Predicted')

axes[1].set_title('TensorFlow')



plt.show()

plt.close()
penalty_l2 = tf.reduce_mean(tf.square(T))

cost_ridge = tf.add(cost, penalty_l2)
with tf.Session() as sess:



    tf.global_variables_initializer().run()



    cost_ridge_history = np.empty(shape=[1], dtype=float)

    t_ridge_history = np.empty(shape=[n_dim, 1], dtype=float)

    

    for epoch in range(training_epochs):



        sess.run(

            training_step,

            feed_dict={P: P_train, q: q_train}

        )

        

        cost_ridge_history = np.append(

            cost_ridge_history,

            sess.run(cost_ridge, feed_dict={P: P_train, q: q_train})

        )

        

        t_ridge_history = np.append(

            t_ridge_history,

            sess.run(T, feed_dict={P: P_train, q: q_train}),

            axis=1

        )

    

    q_pred = sess.run(q_, feed_dict={P: P_test})[:, 0]

    

    mse = tf.reduce_mean(tf.square(q_pred - q_test))

    

    

    

    sess.close()
fig, axes = plt.subplots()



plt.plot(range(len(cost_history)),

         cost_history,

         color='blue', alpha=0.5,

         label='linear regression'

        )



plt.plot(range(len(cost_ridge_history)),

         cost_ridge_history,

         color='red', alpha=0.5,

         label='ridge regression'

        )



axes.set_xlim(xmin=0.95)

axes.set_ylim(ymin=1.e-2)



axes.set_xscale("log", nonposx='clip')

axes.set_yscale("log", nonposy='clip')



axes.set_ylabel('Cost')

axes.set_xlabel(r'Iterations')



axes.legend(loc='upper right')



axes.set_title('Learning rate = ' + str(learning_rate))



plt.show()

plt.close()
# Print again result obtained with TensorFlow & linear regression



print(' --- TensorFlow - linear regression ---')

print(learnt_df[['IMDB_norm_actual', 'IMDB_norm_predicted']].head())



# NOW, TensorFlow & ridge regression



del learnt_df



learnt_df = X_test.copy(deep=True)



learnt_df.insert(loc=0,

                 column='IMDB_norm_predicted',

                 value=pd.Series(data=q_pred, index=learnt_df.index)

                )



learnt_df.insert(loc=0,

                 column='IMDB_norm_actual',

                 value=q_test

                )



print('')

print(' --- TensorFlow - ridge regression ---')

print(learnt_df[['IMDB_norm_actual', 'IMDB_norm_predicted']].head())
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8.,6.))

plt.subplots_adjust(hspace=0.4, wspace=0.4)



panels = [(0, 0), (0, 1), (1, 0), (1, 1)]



for j in range(len(panels)):

  

    axes[panels[j]].plot(t_ridge_history[j][1:],

                    cost_ridge_history[1:],

                    color='red', alpha=0.5,

                    label='ridge regression'

                   )

    

    axes[panels[j]].scatter(t_ridge_history[j][-1],

                    cost_ridge_history[-1],

                    color='black', marker='*',

                    s=100, label='last point'

                   )

        

    axes[panels[j]].plot(t_history[j][1:],

                    cost_history[1:],

                    color='blue', alpha=0.5,

                    label='linear regression'

                   )

        

    axes[panels[j]].scatter(t_history[j][-1],

                    cost_history[-1],

                    color='black', marker='*',

                    s=100

                   )



    axes[panels[j]].set_xlabel(feature_cols[j] + ' coeff.')

    axes[panels[j]].set_ylabel('Cost')



    # axes[panels[j]].set_yscale("log", nonposx='clip')

    

axes[0, 1].legend(bbox_to_anchor=(1, 0.5))

axes[0, 0].set_ylim([0.01, 0.5])

axes[0, 0].set_ylim([-0.1, 0.5])



plt.show()

plt.close()
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10.,6.))

plt.subplots_adjust(hspace=0.4)



axes[0].plot(t_history[0],

          t_history[1],

          color='blue', alpha=0.5,

          label='linear regression'

         )



axes[0].plot(t_ridge_history[0],

          t_ridge_history[1],

          color='red', alpha=0.5,

          label='ridge regression'

         )



axes[0].scatter(t_ridge_history[0][-1],

             t_ridge_history[1][-1],

             color='black', marker='*',

             s=100, zorder=10,

             label='last point'

             )



axes[0].set_xlabel(feature_cols[0] + ' coeff.')

axes[0].set_ylabel(feature_cols[1] + ' coeff.')



axes[0].set_xscale("log", nonposx='clip')

# axes[0].set_yscale("log", nonposy='clip')



axes[0].legend(loc='upper left')



axes[0].set_title('Optimization trajectories on the parameter space')



axes[1].plot(t_history[2],

          t_history[3],

          color='blue', alpha=0.5

          )



axes[1].plot(t_ridge_history[2],

          t_ridge_history[3],

          color='red', alpha=0.5

          )



axes[1].scatter(t_ridge_history[2][-1],

             t_ridge_history[3][-1],

             color='black', marker='*',

             s=100, zorder=10,

             )



axes[1].set_xlabel(feature_cols[2] + ' coeff.')

axes[1].set_ylabel(feature_cols[3] + ' coeff.')



axes[1].set_xscale("log", nonposx='clip')

axes[1].set_yscale("log", nonposy='clip')



plt.show()

plt.close()
fig = plt.figure(figsize=(10.,6.))



axes = fig.add_subplot(111, projection='3d')



axes.scatter(xs=np.log10(t_ridge_history[0][4:]),

             ys=t_ridge_history[1][4:],

             zs=cost_ridge_history[4:],

             zdir='z',

             s=20, 

             color='red',

             depthshade=True

             )



axes.set_xlabel('log_10(' + feature_cols[0] + ' coeff.)', labelpad=17)

axes.set_ylabel(feature_cols[1] + ' coeff.', labelpad=17)

axes.set_zlabel('Cost', labelpad=17, rotation=90)



axes.set_title('Optimization trajectory on the cost hypersurface')



plt.tight_layout()



plt.show()

plt.close()
fig = plt.figure(figsize=(10.,6.))



axes = fig.add_subplot(111, projection='3d')



axes.scatter(xs=np.log10(t_ridge_history[2][4:]),

             ys=np.log10(t_ridge_history[3][4:]),

             zs=cost_ridge_history[4:],

             zdir='z',

             s=20, 

             color='red',

             depthshade=True

             )



axes.set_xlabel('log_10(' + feature_cols[2] + ' coeff.)', labelpad=17)

axes.set_ylabel('log_10(' + feature_cols[3] + ' coeff.)', labelpad=17)

axes.set_zlabel('Cost', labelpad=17, rotation=90)



axes.set_title('Optimization trajectory on the cost hypersurface')



plt.tight_layout()



plt.show()

plt.close()