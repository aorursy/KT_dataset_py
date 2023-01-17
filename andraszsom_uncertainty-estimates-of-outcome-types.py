# Calculate confidence intervals

import scipy.stats
import numpy as np
    
def multinomial_CI(obs=[1,1],conf_level=0.95):
    """
    Calculate confidence intervals for multinomial distributions using Wilson score intervals:
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    
    Parameters
    ----------
    
    obs : list or numpy array with length n
        The observed occurences of each outcome.
    
    conf_level : float between 0 and 1
        The desired confidence level of the confidence interval. 0.95 is a 95% confidence level meaning that 
        the true outcome probability for which the difference between the true probability and the observed 
        rate is not statistically significant at the 5% level.
    
    Returns
    -------
    
    CI : numpy array with shape (n,2)
        The confidence intervales for each outcome. CI[:,0] is the lower end and CI[:,1] is the higher end.
    
    """
    alpha = (1e0-conf_level)
    z = scipy.stats.norm.ppf(1e0-alpha/2e0)
    n_tot = np.sum(obs)
    # observed probabilities
    probs = 1e0*np.array(obs) / n_tot
    
    CI = np.zeros([len(obs),2])
    
    term_pm = z * np.sqrt(probs*(1e0-probs)/n_tot + z**2/(4e0*n_tot**2))
    
    CI[:,0] = (probs + z**2/(2e0*n_tot) - term_pm) / (1e0 + z**2/n_tot)
    CI[:,1] = (probs + z**2/(2e0*n_tot) + term_pm) / (1e0 + z**2/n_tot)
        
    return CI

# Confidence intervals as a function of total counts if one outcome type is counted 0 times

n_sum = [10,30,100,300,1000]

for i in range(len(n_sum)):
    print('Total number of animals in the category:',n_sum[i])
    print('   The corresponding confidence interval of an outcome type with 0 animals is 0 -',np.around(multinomial_CI([0,n_sum[i]])[0,1]*100,2),'%')
     

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

# some outcomes:
# There are roughly 20, 200, and 2000 animals within these breeds (mixes are included!)
vizsla = [ 15.,   0.,   0.,   5.,   0.]
shih_tzu = [  22.,    0.,    5.,   57.,  106.]
pit_bull = [ 737.,    8.,  313.,  750.,  584.]

unique_outcomes = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']

def generate_wedges(data,colors,n_wedges,CI):

    angles = data / np.sum(data) * 360
    cumsum = np.cumsum(angles)
    cumsum = np.insert(cumsum,0,0)
    # the main wedges based on the observed rates, no facecolor just edges
    patches_border = []
    for i in range(len(data)):
        patches_border.append(Wedge((0,0),1,cumsum[i]+90,cumsum[i+1]+90,width=0.5,facecolor='none',edgecolor='k'))

    # lower CIs, no alpha    
    patches_low_CI = []
    for i in range(len(data)):
        wedge_center = (cumsum[i]+cumsum[i+1])/2e0+90
        patches_low_CI.append(Wedge((0,0),1,wedge_center - CI[i,0]/2*360,wedge_center + CI[i,0]/2*360,width=0.5,facecolor=colors[i],edgecolor='none',alpha=1))


    # add n_wedges small Wedges with succesively reduced alphas
    patches_transition = []
    for i in range(len(data)):
        wedge_center = (cumsum[i]+cumsum[i+1])/2e0+90
        step_angle_left = np.linspace(wedge_center - CI[i,0]/2*360,wedge_center - CI[i,1]/2*360,num=n_wedges+1,endpoint=True)
        step_angle_right = np.linspace(wedge_center+CI[i,0]/2*360,wedge_center+CI[i,1]/2*360,num=n_wedges+1,endpoint=True)
        step_alpha = np.linspace(1,0,num=n_wedges,endpoint=False)
        for j in range(n_wedges):

            patches_transition.append(Wedge((0,0),1,step_angle_left[j+1],step_angle_left[j],width=0.5,facecolor=colors[i],edgecolor='none',alpha=step_alpha[j]))
            patches_transition.append(Wedge((0,0),1,step_angle_right[j],step_angle_right[j+1],width=0.5,facecolor=colors[i],edgecolor='none',alpha=step_alpha[j]))
    
    return patches_border,patches_low_CI,patches_transition

colors=['#5A8F29', 'k', '#FF8F00', '#FFF5EE', '#3C7DC4']
n_wedges = 30

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(131, aspect='equal')
plt.title('Vizsla, '+str(int(np.sum(vizsla)))+' animals')
# calculate confidence intervals
CI = multinomial_CI(vizsla)
# generate wedges
patches_border,patches_low_CI,patches_transition = generate_wedges(vizsla,colors,n_wedges,CI)
# plot
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.axis('off')
for p in patches_transition:
    ax1.add_patch(p)
legend = []
for p in patches_low_CI:
    wedge = ax1.add_patch(p)
    legend.append(wedge)
for p in patches_border:
    ax1.add_patch(p)

ax2 = fig.add_subplot(132, aspect='equal')
plt.title('Shih Tzu, '+str(int(np.sum(shih_tzu)))+' animals')
# calculate confidence intervals
CI = multinomial_CI(shih_tzu)
# generate wedges
patches_border,patches_low_CI,patches_transition = generate_wedges(shih_tzu,colors,n_wedges,CI)
#plot
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.axis('off')
for p in patches_transition:
    ax2.add_patch(p)
legend = []
for p in patches_low_CI:
    wedge = ax2.add_patch(p)
    legend.append(wedge)
for p in patches_border:
    ax2.add_patch(p)

ax3 = fig.add_subplot(133, aspect='equal')
plt.title('Pit Bull, '+str(int(np.sum(pit_bull)))+' animals')
# calculate confidence intervals
CI = multinomial_CI(pit_bull)
# generate wedges
patches_border,patches_low_CI,patches_transition = generate_wedges(pit_bull,colors,n_wedges,CI)
#plot
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.axis('off')

legend = []
for p in patches_low_CI:
    wedge = ax3.add_patch(p)
    legend.append(wedge)
for p in patches_transition:
    ax3.add_patch(p)
for p in patches_border:
    ax3.add_patch(p)
    
ax2.legend(legend,unique_outcomes,loc='center',fontsize=12,bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=True, shadow=True)
    
plt.tight_layout(w_pad=4)
plt.savefig('outcome_uncertainty.jpg',dpi=150)
plt.show()

CI_vizsla = np.around(multinomial_CI(vizsla),2)*100
CI_shih_tzu = np.around(multinomial_CI(shih_tzu),2)*100
CI_pit_bull = np.around(multinomial_CI(pit_bull),3)*100

print('Vizsla')
print('        Lower end of CI, observed probability, higher end of CI [%]')
vizsla_probs = vizsla / np.sum(vizsla)*100
for i in range((len(unique_outcomes))):
    print('  ',unique_outcomes[i],':',CI_vizsla[i,0],',',vizsla_probs[i],',',CI_vizsla[i,1])
    
print('Shih Tzu')
print('        Lower end of CI, observed probability, higher end of CI [%]')
shih_tzu_probs = np.around(shih_tzu / np.sum(shih_tzu),2)*100
for i in range((len(unique_outcomes))):
    print('  ',unique_outcomes[i],':',CI_shih_tzu[i,0],',',shih_tzu_probs[i],',',CI_shih_tzu[i,1])
    
print('Pit Bull')
print('        Lower end of CI, observed probability, higher end of CI [%]')
pit_bull_probs = np.around(pit_bull / np.sum(pit_bull),3)*100
for i in range((len(unique_outcomes))):
    print('  ',unique_outcomes[i],':',CI_pit_bull[i,0],',',pit_bull_probs[i],',',CI_pit_bull[i,1])