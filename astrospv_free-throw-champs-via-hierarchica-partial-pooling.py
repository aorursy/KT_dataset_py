#Code here to take the average
%matplotlib inline
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns #pretty plotting

df = pd.read_csv('../input/free_throws.csv')
global_average=float(df['shot_made'].sum())/df['shot_made'].count() #Could also say df['shot_made'].mean() here


print("Average for all players: p={:.3f}".format(global_average))


#Plot the stats for each player, as well as this average

player_grouping=df.groupby(['player'])

shots_made=player_grouping['shot_made'].sum().values.astype(float)
attempts=player_grouping['shot_made'].count().values.astype(float)

names=np.array([name for name, _ in player_grouping['player']])

#Plot it all
with plt.style.context(('seaborn')):
    fig, ax=plt.subplots(figsize=(10, 10))
    scatter=ax.scatter(attempts, shots_made/attempts, c=np.log10(attempts), cmap='plasma')
    ax.axhline(global_average, c='k', linewidth=2.0, linestyle='dashed', label='Global Average')
    cb=fig.colorbar(scatter, ax=ax)
    cb.set_label(r'$\log_{10}$ Attempts', fontsize=20)
    
    ax.legend(fontsize=20, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xscale('log')
    ax.set_xlabel('Attempts', fontsize=20)
    ax.set_ylabel('Conversion Rate', fontsize=20)
    ax.set_title('NBA Free Throws', fontsize=30)
import pymc3 as pm, theano.tensor as tt

#Number of different players we have
N_players=len(attempts)

with pm.Model() as model:
    
    #Hyper parameters on the global skill probability distibution
    phi = pm.Uniform('phi', lower=0.0, upper=1.0)

    kappa_log = pm.Exponential('kappa_log', lam=1.5)
    kappa = pm.Deterministic('kappa', tt.exp(kappa_log))
    
    #Here are the individual beta distributions for each player. We start them off at their values based on
    #the values of phi and kappa, but these will be updated as they see the data.    
    #We can either describe a beta distribution by two variables alpha and beta, or by its mean and standard deviation.
    #Here we have alpha=phi*kappa, and beta=kappa*(1-phi). This implies (via some maths) that phi is the mean of the 
    #beta distribution and kappa is related to the variance. 
    #See http://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html for more infomation!
    individual_player_skills =pm.Beta('individual_player_skills', alpha=phi*kappa, beta=(1.0-phi)*kappa, shape=N_players)
    
    #Our binomial likelihood function
    likelihood = pm.Binomial('likelihood', n=attempts, p=individual_player_skills, observed=shots_made)

    trace = pm.sample(2000, tune=1000, chains=2)

_=pm.traceplot(trace, varnames=['phi', 'kappa'])
#get the skill level assigned to each player
skill_levels=np.mean(trace['individual_player_skills'], axis=0)

#Get a KDE of the global skill and kappa traces- we'll use these in a bit
from scipy import stats
xs_kappa=np.linspace(15.0, 30.0, 1000)
kde_kappa=stats.gaussian_kde(trace['kappa'])
kde_vals_kappa=kde_kappa(xs_kappa)

xs_skill=np.linspace(0.70, 0.74, 1000)
kde_skill=stats.gaussian_kde(trace['phi'])
kde_vals_skill=kde_skill(xs_skill)


#Plot it all
with plt.style.context(('seaborn')):
    fig, ax=plt.subplots(figsize=(10, 10))
    scatter=ax.scatter(attempts, skill_levels, c=np.log10(attempts), cmap='plasma')
    ax.axhline(global_average, c='k', linewidth=2.0, linestyle='dashed', label='Global Average')
    cb=fig.colorbar(scatter, ax=ax)
    cb.set_label(r'$\log_{10}$ Attempts', fontsize=20)
    
    #Details
    ax.legend(fontsize=20, loc='lower left')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xscale('log')
    ax.set_xlabel('Attempts', fontsize=20)
    ax.set_ylabel('Conversion Rate', fontsize=20)
    ax.set_title('NBA Free Throws', fontsize=30)
    ax.set_xlim(0.9, 10**4)
    ax.set_ylim(0.0, 1.0)
    
    # Shade the peak of the Beta distribution (plus uncertainty)
    x=np.linspace(0.1, 2*10**4)
    for i in range(0, 100,  1):
        inds=np.where(kde_vals_kappa>np.percentile(kde_vals_kappa, i))[0]    
        plt.fill_between(x, xs_skill[inds[0]], xs_skill[inds[-1]], alpha=0.8/100.0, facecolor='k')

#Plot of the underlying distribution of player skills

#These are the most probable parameters- the peak of the histograms for each parameter
gskill_map=xs_skill[np.argmax(kde_vals_skill)]
kapp_map=xs_kappa[np.argmax(kde_vals_kappa)]
beta_map=stats.beta(a=gskill_map*kapp_map, b=(1.0-gskill_map)*kapp_map)
#Plot them
with plt.style.context(('seaborn')):
    x=np.linspace(0.0, 1.0, 1000)
    fog, ax=plt.subplots(figsize=(10, 10))
    ax.plot(x, beta_map.pdf(x), c='r', zorder=10, label='Probability distribution\nof player skill')

    #Draw random samples from our chain to get an idea of the uncertainity
    randoms=np.random.randint(0, len(trace['phi']), size=1000)
    for gskill, kapp in zip(trace['phi'][randoms], trace['kappa'][randoms]):

        beta=stats.beta(a=gskill*kapp, b=(1.0-gskill)*kapp)
        ax.plot(x, beta.pdf(x), c='k', alpha=0.1)

    #Add a histogram of the original shots data
    hist=ax.hist(shots_made/attempts.astype(float), 50, normed=True, facecolor='b', alpha=0.8, label='Original data: \nshots/attempts')  

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel(r'$p_{i}$', fontsize=20)
    ax.legend(loc='upper left', fontsize=20)


#The indices which would sort the array of player skills
top_inds=np.argsort(skill_levels)[-10:]
bottom_inds=np.argsort(skill_levels)[:10]
inds=np.concatenate((bottom_inds, top_inds))

#Traces and names
t=trace['individual_player_skills'][:, inds]
n=names[inds]
n=np.insert(n, 10, [''])

#Make a violin plot
with plt.style.context(('seaborn')):
    fig, ax=plt.subplots(figsize=(10, 10))
    parts=ax.violinplot(t, positions=np.delete(np.arange(21), 10), vert=False, showextrema=True, showmedians=True)
    ax.axhline(10.0, linestyle='dashed', linewidth=2.0, c='k')
    #Details
    ax.set_yticks(np.arange(0, 21))
    ax.set_xticks(np.arange(1, 11)/10.)
    ax.set_yticklabels(list(n), fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(r'p_i')
    #Colour the violins by log10(attempts)
    cm=plt.get_cmap('plasma')
    for pc, ind in zip(parts['bodies'], inds):
        c=cm(np.log10(attempts[ind])/np.max(np.log10(attempts)))
        pc.set_facecolor(c)
        pc.set_edgecolor('k')
        pc.set_alpha(1)
    #Reuse the same colorbar from before
    cb=fig.colorbar(scatter, ax=ax)
    cb.set_label(r'$\log_{10}$ Attempts', fontsize=20)
    

