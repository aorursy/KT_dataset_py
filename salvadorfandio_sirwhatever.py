import io

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.optimize

from math import erf, log, exp, sqrt

from scipy.stats import invgauss
def lognorm_cdf(mean, var, n_days=20, ticks_per_day=10):

    sigma = sqrt(log(var/mean**2 + 1))

    mu = log(mean) - 0.5*sigma**2

    sqrt2sigma_inv = 1.0 / (sqrt(2) * sigma)

    def cdf(x):

        return (0.5 + 0.5*erf(sqrt2sigma_inv*(log(x) - mu))

                if x > 0 else 0)



    n_ticks = n_days * ticks_per_day

    days = list([i/ticks_per_day

                 for i in range(n_ticks)])

    acu = list([cdf(d) for d in days])

    return acu
def invgauss_cdf(mean, var, n_days=20, ticks_per_day=10):

    la = mean**3/var

    mu = mean/la

    x = list([i/ticks_per_day

              for i in range(n_days*ticks_per_day)])

    return invgauss.cdf(x, mu, scale=la)
class SIRwhatever:

    def __init__(self,

                 population_size=6.662e6,

                 asymptomatic_ratio=0.17,

                 severe_ratio=0.05,

                 #hospital_beds_ratio=0.003,

                 #no_bed_death_rate_multiplier=2,

                 mortality_ratio=0.035,

                 asymptomatic_convalescent_period_mean=2,

                 asymptomatic_convalescent_period_variance=3,

                 mild_convalescent_period_mean=8,

                 mild_convalescent_period_variance=4,

                 severe_convalescent_period_mean=16,

                 severe_convalescent_period_variance=8,

                 daily_transmission_rate=2.0,

                 preinfectious_half_period=2,

                 incubation_mean=5,

                 incubation_variance=7,

                 herd_immunity_ratio=0.0,

                 days = 100, ticks_per_day = 4,

                 infected_seed = 10,

                 observation_delay = 0,

                 quiet = False):

        self.population_size = population_size

        self.asymptomatic_ratio = asymptomatic_ratio

        self.severe_ratio = severe_ratio

        self.mortality_ratio = mortality_ratio

        self.daily_transmission_rate = daily_transmission_rate

        self.days = days

        self.ticks_per_day = ticks_per_day

        self.infected_seed = infected_seed

        self.incubation_mean = incubation_mean

        self.incubation_variance = incubation_variance

        self.preinfectious_half_period = preinfectious_half_period

        self.mild_convalescent_period_mean = mild_convalescent_period_mean

        self.mild_convalescent_period_variance = mild_convalescent_period_variance

        self.severe_convalescent_period_mean = severe_convalescent_period_mean

        self.severe_convalescent_period_variance = severe_convalescent_period_variance

        self.asymptomatic_convalescent_period_mean = asymptomatic_convalescent_period_mean

        self.asymptomatic_convalescent_period_variance = asymptomatic_convalescent_period_variance

        self.herd_immunity_ratio=herd_immunity_ratio

        self.observation_delay=observation_delay

        self.quiet=quiet

    

    def strategy(self, t, **kargs):

        return 'normal' # can I do nothing?

    

    def print_extrema(self, i, t, **kargs):

        if self.quiet:

            return

        if t >= 2: # we ignore the first two days in order to let the model stabilize

            for k in sorted(kargs.keys()):

                extreme = None

                v = kargs[k]

                if v[i-2] < v[i-1]:

                    if v[i-1] > v[i]:

                        extreme = '▲'

                elif (v[i-1] < v[i]) and (v[i-2] > v[i-1]):

                    extreme = '▼'

                if extreme:

                    line = "%s day: %f, %s: %f" % (extreme, t, k, v[i-1])

                    if k.startswith('new_'):

                        line += ", %s: %f" % (k[4:], kargs[k[4:]][i-1])

                    if ("ever_" + k) in kargs.keys():

                        line += ", ever_%s: %f" % (k, kargs["ever_" + k][i-1])

                    print(line)

    

    def run(self):

        ticks = self.days * self.ticks_per_day # loop iterations

        Dt = 1 / self.ticks_per_day

        

        z = np.zeros(ticks)

        

        healthy = z.copy()              # people that is not infected

        death = z.copy()                # acumulated deaths

        incubating = z.copy()           # infected in the incubation stage

        infected = z.copy()             # number of currently infected persons

        velocity = z.copy()             # contagion velocity

        sick = z.copy()                 # people with desease simptoms

        new_sick = z.copy()             # new people getting sick at the tick

        sick_velocity = z.copy()        # velocity of sick cases appearing

        ever_sick = z.copy()            # people that has ever been sick

        ever_sick_velocity = z.copy()   # velocity of ever sick

        severe = z.copy()               # people in severe condition

        mild = z.copy()                 # people in mild condition

        asymptomatic = z.copy()         # infected people not showing symptoms

        infectious = z.copy()           # infected people that is able to transmit the disease

        infectable = z.copy()           # people that can still get the dissease

        symptomatic = z.copy()          # sick people showing symptoms

        new_symptomatic = z.copy()      # new people getting symptoms at the tick

        symptomatic_velocity = z.copy() # velocity of symptomatic cases appearing

        ever_symptomatic = z.copy()     # people that have ever had symptoms

        ever_symptomatic_velocity = z.copy() # velocity of ever symptomatic cases appearing

        observed_ever_symptomatic = z.copy() # people that has ever been symptomatic and observed

        observed_ever_symptomatic_velocity = z.copy() # velocity of observed ever symptomatic



        # new_sick_velocity = z.copy()    # velocity of new sick

        day = z.copy()

        immune = np.full(ticks,        # people that has healed from the disease 

                         self.herd_immunity_ratio*self.population_size)



        status = list(['normal' for i in range(ticks)])



        

        # Here the probability distributions are tabled and cached.

        incubation_cdf = lognorm_cdf(self.incubation_mean, self.incubation_variance,

                                     ticks_per_day = self.ticks_per_day, n_days=self.days)

        mild_convalescent_period_cdf = invgauss_cdf(self.mild_convalescent_period_mean,

                                                    self.mild_convalescent_period_variance,

                                                    ticks_per_day = self.ticks_per_day,

                                                    n_days=self.days)

        severe_convalescent_period_cdf = invgauss_cdf(self.severe_convalescent_period_mean,

                                                      self.severe_convalescent_period_variance,

                                                      ticks_per_day = self.ticks_per_day,

                                                      n_days=self.days)

        asymptomatic_convalescent_period_cdf = invgauss_cdf(self.asymptomatic_convalescent_period_mean,

                                                            self.asymptomatic_convalescent_period_variance,

                                                            ticks_per_day = self.ticks_per_day,

                                                            n_days=self.days)

        

        preinfection_amortiguation_rate = 0.5**(1.0/(self.preinfectious_half_period

                                                     * self.ticks_per_day))

        

        for i in range(ticks):



            ps = self.population_size

            t = i * Dt

            day[i] = t # For convenience, we also store the time

            

            # Number of coupons an infectious person generates at his maximum

            transmission_rate = self.daily_transmission_rate / self.ticks_per_day

            

            # Every infectious person gives away a number of coupons

            # everyday (actually, every tick)

            coupons = infectious[i] * transmission_rate

            

            # A single player may get several coupons, so we need to calculate the probability

            # that has a player of at least getting a coupon. We do it calculating first the

            # probability of not getting any coupon at all.

            p_contagied = 1 - exp(-coupons/ps) # We can calculate it using an aproximation

                                               # in order to avoid numerical inestabilities

                                               # It works because ps is quite big 

            # p_contagied = 1 - ((ps - 1)/ps) ** coupons # probability of somebody

            #                                            # not getting any ticket

            

            # And how many infectable persons do we have to play today?

            players = ps

            if i == 0:

                d_infected = self.infected_seed

            else:

                players -= infected[i-1] + immune[i-1] + death[i-1]

                d_infected = players * p_contagied # number of persons that have

                                                   # been infected today

            infectable[i] = players - d_infected

            

            # Spread the infected over the following days, recording when they are going

            # to get sick filling new_sick

            for j in range(i+1, ticks):

                t1 = j - i # time elapsed since the contagion

                cdf0 = incubation_cdf[t1 - 1]

                if cdf0 == 1.0:

                    break

                cdf1 = incubation_cdf[t1]

                d_new_sick = d_infected * (cdf1 - cdf0)

                new_sick[j] += d_new_sick

                incubating[j-1] += d_infected*(1-cdf0)



                # Here we also fill infectious which counts how many people is able

                # to generate coupons

                for k in range(j, i+1, -1):

                    preinfection_rate = preinfection_amortiguation_rate**(j-k) * (k-i)/(j-i)

                    if preinfection_rate < 1e-4:

                        break

                    infectious[k] += d_new_sick * preinfection_rate

            

            # Now, we see what happens which people getting sick right now.

            new_sick_i = new_sick[i]

            

            # How many is there in every class?

            d_asymptomatic = new_sick_i * self.asymptomatic_ratio

            d_severe = new_sick_i * self.severe_ratio

            d_mild = new_sick_i - d_asymptomatic - d_severe

            new_symptomatic[i] = new_sick_i - d_asymptomatic

            

            # Only the severyly ill can die, so we calculate and adjusted

            # mortality ratio just for those

            severe_mortality_ratio = self.mortality_ratio / self.severe_ratio 



            # The future of those getting ill today is set in stone, so we can fill

            # the asymptomatic, mild, severe, imune, death until the end of time for them.

            for j in range(i, ticks):

                t1 = j - i

                asymptomatic_cdf0=asymptomatic_convalescent_period_cdf[t1]

                mild_cdf0=mild_convalescent_period_cdf[t1]

                severe_cdf0=severe_convalescent_period_cdf[t1]

                severe[j] += (1-severe_cdf0)*d_severe

                mild[j] += (1-mild_cdf0)*d_mild

                asymptomatic[j] += (1-asymptomatic_cdf0)*d_asymptomatic

                

                immune[j] += ((1-severe_mortality_ratio)*severe_cdf0*d_severe +

                              mild_cdf0*d_mild + asymptomatic_cdf0*d_asymptomatic)

                death[j] += severe_mortality_ratio*severe_cdf0*d_severe

                infectious[j] += (1-asymptomatic_cdf0)*d_asymptomatic



            # infected, sick and healthy are just combinations of other parameters

            # we keep for convenience.

            infected[i] = asymptomatic[i] + mild[i] + severe[i] + incubating[i]

            symptomatic[i] = mild[i] + severe[i]

            sick[i] = symptomatic[i] + asymptomatic[i]

            healthy[i] = infectable[i] + immune[i]

            

            # Finally, also for convenience, we calculate the velocities for some parameters.

            # In this context, velocity is the increase per time unit of some parameter in

            # the exponencial space:

            #     velocity(a) = exp(log(a[i]) - log(a[i-1])) = a[i] / a[i-1]

            

            if i == 0:

                velocity[i] = 0

                sick_velocity[i] = 0

                ever_sick[i] = new_sick[0]

                ever_sick_velocity[i] = 0

                ever_symptomatic[i] = 0

            else:

                velocity[i] = ((infected[i]/infected[i-1])**self.ticks_per_day - 1)

                sick_velocity[i] = (((sick[i]+1)/(sick[i-1]+1))**self.ticks_per_day - 1)

                ever_sick[i] = ever_sick[i-1] + new_sick[i]

                ever_sick_velocity[i] = (((ever_sick[i] + 1) /

                                          (ever_sick[i-1] + 1))**self.ticks_per_day - 1)

                # new_sick_velocity[i] = (((new_sick[i] + 1) /

                #                          (new_sick[i-1] + 1))**self.ticks_per_day - 1)

                symptomatic_velocity[i] = (((symptomatic[i]+1)/(symptomatic[i-1]+1)) **

                                           self.ticks_per_day - 1)

                ever_symptomatic[i] = ever_symptomatic[i-1] + new_symptomatic[i]

                ever_symptomatic_velocity[i] = (((ever_symptomatic[i]+1) /

                                                 (ever_symptomatic[i-1]+1)) **

                                                self.ticks_per_day - 1)

                

                observation_delay_ticks = self.observation_delay * self.ticks_per_day

                if i >= observation_delay_ticks:

                    observed_ever_symptomatic[i] = ever_symptomatic[i-observation_delay_ticks]

                    observed_ever_symptomatic_velocity[i] = ever_symptomatic_velocity[i-observation_delay_ticks]

            

            self.print_extrema(i, t=t,

                               sick=sick,

                               new_sick=new_sick,

                               ever_sick=ever_sick,

                               severe=severe,

                               symptomatic=symptomatic,

                               ever_symptomatic=ever_symptomatic,

                               new_symptomatic=new_symptomatic,

                               asymptomatic=asymptomatic,

                               infected=infected,

                               incubating=incubating,

                               infectious=infectious)

            

            # This method call allows us to simulate external actions over

            # the model: The Goberment!

            status[i] = self.strategy(t=t,

                                      sick=sick[i],

                                      ever_sick=ever_sick[i],

                                      sick_velocity=sick_velocity[i],

                                      new_sick=new_sick[i]*self.ticks_per_day,

                                      symptomatic=symptomatic[i],

                                      ever_symptomatic=ever_symptomatic[i],

                                      observed_ever_symptomatic=observed_ever_symptomatic[i],

                                      symptomatic_velocity=symptomatic_velocity[i],

                                      observed_ever_symptomatic_velocity=observed_ever_symptomatic_velocity[i],

                                      new_symptomatic=new_symptomatic[i]*self.ticks_per_day,

                                      infectious=infectious[i],

                                      infectable=infectable[i])

            if i > 0 and status[i] != status[i-1] and not self.quiet:

                print(("status flip: %s -> %s, day: %f, " +

                       "sick: %f, daily_new_sick: %f, ever_sick: %f, "+

                       "symptomatic: %f, daily_new_symptomatic: %f, ever_symptomatic: %f, "+

                       "observed_ever_symptomatic: %f, "+

                       "infectious: %f, severe: %f, immune: %f, death: %f") %

                      (status[i-1], status[i], t,

                       sick[i], new_sick[i]*self.ticks_per_day, ever_sick[i],

                       symptomatic[i], new_symptomatic[i]*self.ticks_per_day,

                       ever_symptomatic[i], observed_ever_symptomatic[i],

                       infectious[i], severe[i], immune[i], death[i]))



        return pd.DataFrame({'day': day,

                             'infectious': infectious,

                             'asymptomatic': asymptomatic,

                             'symptomatic': symptomatic,

                             'mild': mild,

                             'severe': severe,

                             'immune': immune,

                             'infected': infected,

                             'death': death,

                             'healthy': healthy,

                             'incubating': incubating,

                             'sick': sick,

                             'ever_sick': ever_sick,

                             'new_sick': new_sick,

                             'ever_symptomatic': ever_symptomatic,

                             'new_symptomatic': new_symptomatic,

                             'infectable': infectable,

                             'velocity': velocity,

                             'sick_velocity': sick_velocity,

                             'ever_sick_velocity': ever_sick_velocity,

                             'symptomatic_velocity': symptomatic_velocity,

                             'ever_symptomatic_velocity': ever_symptomatic_velocity,

                             'observed_ever_symptomatic': observed_ever_symptomatic,

                             'observed_ever_symptomatic_velocity': observed_ever_symptomatic_velocity,

                             'status': status

                            })
case1 = SIRwhatever(days = 100).run()
with sns.axes_style("whitegrid"):

    case1.plot(x='day', y=['infected', 'sick', 'symptomatic', 'asymptomatic', 'immune',

                           'incubating', 'infectable', 'severe', 'death', 'healthy',

                           'ever_sick', 'observed_ever_symptomatic'],

               logy=False, ylim=(0, 7e6), figsize=(20,10))
with sns.axes_style("whitegrid"):

    case1.plot(x='day', y=['velocity', 'ever_symptomatic_velocity'],

               ylim=(-0.4, 0.7), figsize=(20,10))

    case1.plot(x='day', y=['velocity', 'ever_symptomatic_velocity'],

               ylim=(0, 0.6), xlim=(5,40), figsize=(20,10))
with sns.axes_style("whitegrid"):

    case1.plot(x='day', y=['infected', 'sick', 'immune', 'incubating',

                           'infectable', 'severe', 'death', 'healthy', 'ever_sick'],

               logy=True, ylim=(10, 7e6), figsize=(20,10))
severe_max = case1.severe.max()

public_hospital_beds = 6.662e6 * 0.003

print("max severe: %d, public hospital beds: %d, ratio: %1.1f%%" %

      (severe_max, public_hospital_beds, 100*public_hospital_beds/severe_max))

print("max ever sick: %f, deaths: %f" % (case1.ever_sick.max(), case1.death.max()))
## Implementation of the Isolate N days strategy



class IsolateNDays(SIRwhatever):

    def __init__(self,

                 isolation_days = 15,

                 ever_symptomatic_threshold = 2900,

                 isolation_transmission_rate_divider = 10,

                 after_isolation_transmission_rate_divider = 1,

                 **kargs):

        super().__init__(**kargs)

        self.isolation_days = isolation_days

        self.ever_symptomatic_threshold = ever_symptomatic_threshold

        self.isolation_transmission_rate_divider = isolation_transmission_rate_divider

        self.after_isolation_transmission_rate_divider = after_isolation_transmission_rate_divider

        self.isolated = False

        self.isolated_countdown = 0

        

    def strategy(self, t, observed_ever_symptomatic, **kargs):

        if self.isolated:

            if self.isolated_countdown < 1:

                self.isolated = False

                self.daily_transmission_rate *= (self.isolation_transmission_rate_divider /

                                                 self.after_isolation_transmission_rate_divider)

                self.ever_symptomatic_threshold = self.population_size * 10 # never ever stop

                                                                            # the country, you lazy!

            else:

                self.isolated_countdown -= 1

        else:

            if observed_ever_symptomatic > self.ever_symptomatic_threshold:

                self.isolation_day = t

                self.isolated = True

                self.isolated_countdown = self.isolation_days * self.ticks_per_day

                self.daily_transmission_rate /= self.isolation_transmission_rate_divider

        if self.isolated:

            return 'isolated'

        else:

            return 'normal'
case2 = IsolateNDays(days = 120, isolation_days=1000).run()
with sns.axes_style("whitegrid"):

    case2.plot(x='day', y=['infected', 'sick', 'immune', 'incubating', 'infectable',

                           'symptomatic', 'infectious', 'severe', 'death', 'healthy',

                           'ever_sick'],

               logy=True, ylim=(1, 3e4), figsize=(20,10))

    

    case2.plot(x='day', y=['infected', 'sick', 'immune', 'incubating', 'infectable',

                           'symptomatic', 'infectious', 'severe', 'death', 'healthy',

                           'ever_sick'],

               logy=True, ylim=(1, 1.6e4), xlim=(18, 60), figsize=(20,10))
print("max ever_symptomatic: %d, max symptomatic: %d, max severe: %d, max death: %d" %

      (case2.ever_symptomatic.max(), case2.symptomatic.max(),

       case2.severe.max(), case2.death.max()))
case2b = IsolateNDays(days=300,

                      isolation_transmission_rate_divider=3,

                      isolation_days=1000).run()
with sns.axes_style("whitegrid"):

    case2b.plot(x='day', y=['infected', 'sick', 'immune', 'incubating', 'infectable',

                            'symptomatic', 'infectious', 'severe', 'death', 'healthy',

                            'ever_sick'],

               logy=True, ylim=(10, 7e6), figsize=(20,10))
print("max ever_symptomatic: %d, max symptomatic: %d, max severe: %d" %

      (case2b.ever_symptomatic.max(), case2b.symptomatic.max(), case2b.severe.max()))
case2c = IsolateNDays(days = 70,

                      isolation_transmission_rate_divider = 100,

                      isolation_days=1000).run()
with sns.axes_style("whitegrid"):

    case2c.plot(x='day', y=['infected', 'immune', 'incubating', 'infectable', 'symptomatic',

                            'infectious', 'severe', 'death', 'healthy', 'ever_sick'],

               logy=True, ylim=(1, 2e4), figsize=(20,10))
case2d = IsolateNDays(days=80,

                      isolation_transmission_rate_divider=1e8,

                      isolation_days=1000).run()
with sns.axes_style("whitegrid"):

    case2d.plot(x='day', y=['infected', 'symptomatic', 'immune', 'incubating', 'infectable',

                            'infectious', 'severe', 'death', 'healthy', 'ever_sick'],

               logy=True, ylim=(1, 2e4), figsize=(20,10))
with sns.axes_style("whitegrid"):

    case2d.plot(x='day', y=['velocity', 'ever_symptomatic_velocity'],

                ylim=(-0.4, 0.6), figsize=(20,10))
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set(ylim=(-0.6, 0.5))

    case2.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

               label='infectious case 2')

    case2b.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                label='infectious case 2b')

    case2c.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                label='infectious case 2c')

    case2d.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                label='infectious case 2d')

    
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set(ylim=(1, 1e4))

    case2.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

               label='infectious case 2')

    case2b.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                label='infectious case 2b')

    case2c.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                label='infectious case 2c')

    case2d.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                label='infectious case 2d')
case3 = IsolateNDays(days = 120, isolation_days=15).run()
with sns.axes_style("whitegrid"):

    case3.plot(x='day', y=['infected', 'symptomatic', 'immune', 'incubating', 'infectable',

                           'infectious', 'severe', 'death', 'healthy', 'ever_sick',

                           'new_sick'],

               logy=True, ylim=(10, 7e6), figsize=(20,10))

    case3.plot(x='day', y=['infected', 'symptomatic', 'immune', 'incubating', 'infectable',

                           'infectious', 'severe', 'death', 'healthy', 'ever_sick',

                           'new_sick'],

               logy=False, ylim=(10, 2e4), xlim=(18, 50), figsize=(20,10))
case3b = IsolateNDays(days = 120, isolation_days=30).run()
with sns.axes_style("whitegrid"):

    case3b.plot(x='day', y=['infected', 'symptomatic', 'immune', 'incubating', 'infectable',

                            'severe', 'death', 'healthy', 'ever_sick', 'infectious'],

                logy=True, ylim=(10, 7e6), figsize=(20,10))
case3c = IsolateNDays(days = 150, isolation_days=45,

                      isolation_transmission_rate_divider=10,

                      after_isolation_transmission_rate_divider=3).run()

with sns.axes_style("whitegrid"):

    case3c.plot(x='day', y=['infected', 'symptomatic', 'immune', 'incubating', 'infectable',

                            'severe', 'death', 'healthy', 'ever_sick', 'infectious'],

                logy=True, ylim=(10, 7e6), figsize=(20,10))
case3d2 = IsolateNDays(days = 120, isolation_days=30,

                       isolation_transmission_rate_divider=10,

                       after_isolation_transmission_rate_divider=2).run()

case3d4 = IsolateNDays(days = 120, isolation_days=30,

                       isolation_transmission_rate_divider=10,

                       after_isolation_transmission_rate_divider=4).run()

case3d5 = IsolateNDays(days = 120, isolation_days=30,

                       isolation_transmission_rate_divider=10,

                       after_isolation_transmission_rate_divider=5).run()
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set(ylim=(1, 1e4))

    case3b.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                label='after divider 1.0')

    case3c.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                label='after divider 3.0')

    case3d2.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                 label='after divider 2.0')

    case3d4.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                 label='after divider 4.0')

    case3d5.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                 label='after divider 5.0')

    

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set(ylim=(-0.4, 0.6))

    case3b.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                label='after divider 1.0')

    case3c.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                label='after divider 3.0')

    case3d2.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                 label='after divider 2.0')

    case3d4.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                 label='after divider 4.0')

    case3d5.plot(x='day', y='velocity', logy=False, ax=ax, xlim=(15, 70),

                 label='after divider 5.0')

    
class EverybodyIsHealthy(SIRwhatever):

    def __init__(self,

                 isolation_safe_days = 10,

                 isolation_transmission_rate_divider = 10,

                 ever_symptomatic_thresholds = [2900, 50, 10, 10, 1],

                 **kargs):

        super().__init__(**kargs)

        self.isolated = False

        self.isolation_transmission_rate_divider = isolation_transmission_rate_divider

        self.isolation_safe_days = isolation_safe_days

        self.ever_symptomatic_thresholds = ever_symptomatic_thresholds

        self.flips = 0

        self.ever_symptomatic_at_flop = 0

        

    def strategy(self, t, sick, new_symptomatic, infectious, observed_ever_symptomatic, **_):

        ever_symptomatic_threshold_ix = (self.flips

                                         if self.flips < len(self.ever_symptomatic_thresholds)

                                         else -1)

        ever_symptomatic_threshold = self.ever_symptomatic_thresholds[ever_symptomatic_threshold_ix]

        if self.isolated:

            if new_symptomatic < 1:

                self.isolation_safe_days_countdown -= 1

                if self.isolation_safe_days_countdown < 1:

                    self.isolated=False

                    self.daily_transmission_rate *= self.isolation_transmission_rate_divider

                    self.ever_symptomatic_at_flop = observed_ever_symptomatic

        else:

            if (observed_ever_symptomatic - self.ever_symptomatic_at_flop > ever_symptomatic_threshold):

                self.isolated=True

                self.isolation_safe_days_countdown = self.isolation_safe_days

                self.daily_transmission_rate /= self.isolation_transmission_rate_divider

                self.flips += 1



        return ('isolated' if self.isolated else 'normal')
case4 = EverybodyIsHealthy(days = 250).run()
with sns.axes_style("whitegrid"):

    case4.plot(x='day', y=['infected', 'symptomatic', 'immune', 'incubating', 'severe',

                           'asymptomatic', 'ever_sick', 'infectable'],

               logy=False, figsize=(20,10), ylim=(0, 2e4)) 

    case4.plot(x='day', y=['infected', 'symptomatic', 'immune', 'incubating', 'severe',

                           'asymptomatic', 'ever_sick', 'infectable'],

               logy=True, figsize=(20,10), ylim=(1, 2e4))
mortality_table=pd.read_csv(io.StringIO("""

population,start_age,end_age,mortality

704650,0,9,0.0

590337,10,19,0.002

834247,20,29,0.002

1227008,30,39,0.002

1049187,40,49,0.004

782873,50,59,0.013

597152,60,69,0.036

420139,70,79,0.08

284087,80,200,0.148

"""))
class StratifiedSpread(SIRwhatever):

    def __init__(self,

                 rampant_ever_symptomatic_threshold = 2900,

                 isolated_new_symptomatic_threshold = 1,

                 isolated_transmission_rate_divider = 10,

                 countdown_days = 10,

                 stratified_transmission_rate_multiplier = 1,

                 stratified_exclusion_age = 60,

                 mortality_table = mortality_table,

                 stratified_new_symptomatic_threshold = 1,

                 **kargs):

        super().__init__(**kargs)

        self.ss_state = 'rampant'

        self.rampant_ever_symptomatic_threshold = rampant_ever_symptomatic_threshold

        self.isolated_new_symptomatic_threshold = isolated_new_symptomatic_threshold

        self.isolated_transmission_rate_divider = isolated_transmission_rate_divider

        self.countdown_days = countdown_days

        self.stratified_exclusion_age = stratified_exclusion_age

        self.stratified_new_symptomatic_threshold = stratified_new_symptomatic_threshold

        self.mortality_table = mortality_table



        self.strategies = { k: getattr(self, "strategy_" + k)

                           for k in ['rampant', 'isolated', 'countdown',

                                     'stratified', 'everybody'] }

        self.starts     = { k: getattr(self, "start_" + k)

                           for k in ['rampant', 'isolated', 'countdown',

                                     'stratified', 'everybody'] }

        

        # we keep using the data given by the user, so we calculate the scaling

        # factors for the mortality table accordingly:

        mt_population_size = mortality_table.population.sum()

        self.mt_population_multiplier = self.population_size / mt_population_size

        mt_mortality_ratio = ((mortality_table.population * mortality_table.mortality).sum() /

                              mt_population_size)

        self.mt_mortality_multiplier = self.mortality_ratio / mt_mortality_ratio



    def strategy(self, **kargs):

        old_state = self.ss_state

        state_strategy = self.strategies[old_state]

        new_state = state_strategy(**kargs)

        if new_state is None or new_state == old_state:

            return old_state

        self.ss_state = new_state

        state_start = self.starts[new_state]

        state_start(**kargs)

        return new_state

    

    def start_rampant(self, **kargs):

        pass #



    def strategy_rampant(self, observed_ever_symptomatic, **kargs):

        if (observed_ever_symptomatic > self.rampant_ever_symptomatic_threshold):

            return 'isolated'

            

    def start_isolated(self, **kargs):

        self.daily_transmission_rate /= self.isolated_transmission_rate_divider

        

    def strategy_isolated(self, new_symptomatic, **kargs):

        if new_symptomatic < self.isolated_new_symptomatic_threshold:

            return 'countdown'

    

    def start_countdown(self, **kargs):

        self.countdown_ticks = self.countdown_days * self.ticks_per_day

    

    def strategy_countdown(self, **kargs):

        if self.countdown_ticks < 1:

            self.daily_transmission_rate *= self.isolated_transmission_rate_divider

            return 'stratified'

        self.countdown_ticks -= 1

    

    def start_stratified(self, observed_ever_symptomatic, **kargs):

        sxa = self.stratified_exclusion_age

        mt = self.mortality_table

        mt_below_sxa = mt[mt.start_age < sxa]

        mt_over_sxa = mt[mt.end_age > sxa]

        mt_below_sxa_population = mt_below_sxa.population.sum()

        mt_over_sxa_population = mt_over_sxa.population.sum()

        mortality_ratio = ((mt_below_sxa.mortality*mt_below_sxa.population).sum()/

                            mt_below_sxa_population)*self.mt_mortality_multiplier

        multiplier = mortality_ratio/self.mortality_ratio

        self.mortality_ratio *= multiplier

        self.severe_ratio *= multiplier

        self.population_size = (self.population_size * mt_below_sxa_population /

                                (mt_below_sxa_population + mt_over_sxa_population))

        print("multiplier: %f, mortality_ratio: %f, population_size: %f" %

              (multiplier, self.mortality_ratio, self.population_size))

    

    def strategy_stratified(self, new_symptomatic, observed_ever_symptomatic, **kargs):

        if (new_symptomatic < self.stratified_new_symptomatic_threshold and

            observed_ever_symptomatic > 0.33 * self.population_size):

            return 'everybody'

    

    def start_everybody(self, infectable, **kargs):

        sxa = self.stratified_exclusion_age

        mt = self.mortality_table

        mt_below_sxa = mt[mt.start_age < sxa]

        mt_over_sxa = mt[mt.end_age > sxa]

        mt_below_sxa_population = mt_below_sxa.population.sum()

        mt_over_sxa_population = mt_over_sxa.population.sum()

        stratified_population_size = self.population_size

        excluded_population_size = (stratified_population_size *

                                    mt_over_sxa_population / mt_below_sxa_population)

        excluded_population_size_adjusted = excluded_population_size * self.mt_population_multiplier

        stratified_mortality_rate = ((mt_below_sxa.mortality*mt_below_sxa.population).sum()/

                                     mt_below_sxa_population)

        excluded_mortality_rate = ((mt_over_sxa.mortality*mt_over_sxa.population).sum()/

                                     mt_over_sxa_population)

        combined_mortality_rate = ((stratified_mortality_rate * infectable +

                                    excluded_mortality_rate * excluded_population_size_adjusted) /

                                   (infectable + excluded_population_size_adjusted))

        multiplier = combined_mortality_rate / self.mortality_ratio

        

        self.mortality_ratio *= multiplier

        self.severe_ratio *= multiplier

        self.population_size = ((self.population_size / stratified_population_size) *

                                stratified_population_size + excluded_population_size)

        print("multiplier: %f, mortality_ratio: %f, population_size: %f" %

              (multiplier, self.mortality_ratio, self.population_size))

        

    def strategy_everybody(self, **kargs):

        pass
case5 = StratifiedSpread(days=400, stratified_exclusion_age=50).run()
with sns.axes_style("whitegrid"):

    case5.plot(x='day', y=['infected', 'sick', 'immune', 'incubating', 'infectable',

                           'asymptomatic', 'infectious', 'severe', 'death', 'healthy',

                           'ever_sick', 'new_sick'],

               logy=True, ylim=(1, 9e6), figsize=(20,10))

    case5.plot(x='day', y='velocity',

               logy=False, ylim=(-0.3, 0.6), figsize=(20,5))

case5.velocity[case5.day==350]
case5.death[case5.day==173]
case5.severe[case5.day < 173].max()
case5b = StratifiedSpread(days=300,

                          stratified_exclusion_age=60).run()
with sns.axes_style("whitegrid"):

    case5b.plot(x='day', y=['infected', 'sick', 'immune', 'incubating', 'infectable',

                            'infectious', 'severe', 'death', 'healthy', 'ever_sick',

                            'new_sick'],

                logy=True, ylim=(1, 7e6), figsize=(20,10))

    case5b.plot(x='day', y='velocity',

               logy=False, ylim=(-0.3, 0.6), figsize=(20,5))
case5b.death.max()
case5b.severe.max()
case5c = StratifiedSpread(days=300,

                          stratified_exclusion_age=40).run()
with sns.axes_style("whitegrid"):

    case5c.plot(x='day', y=['infected', 'sick', 'immune', 'incubating', 'infectable',

                            'infectious', 'severe', 'death', 'healthy', 'ever_sick',

                            'new_sick'],

                logy=True, ylim=(1, 7e6), figsize=(20,10))

    case5c.plot(x='day', y='velocity',

               logy=False, ylim=(-0.3, 0.6), figsize=(20,5))
case5c[case5c.day==200].velocity
fig, ax = plt.subplots(figsize=(20, 10))

ax.set(ylim=(1, 1e4))



for incubation_variance in [3, 7, 12]:

    for asymptomatic_convalescent_period_variance in [3, 6, 12]:

        df = IsolateNDays(days = 70,

                          isolation_transmission_rate_divider = 10,

                          asymptomatic_convalescent_period_variance=asymptomatic_convalescent_period_variance,

                          incubation_variance=incubation_variance,

                          isolation_days=70,

                          quiet=True).run()

        df.plot(x='day', y='infectious', logy=True, ax=ax, xlim=(15, 70),

                label=('iv: %f, acpv: %f' %

                       (incubation_variance, asymptomatic_convalescent_period_variance)))
fig, axs = plt.subplots(2, figsize=(20, 15))

axs[0].set(ylim=(1, 2e4))

axs[1].set(ylim=(-0.2, 0.45))



for asymptomatic_convalescent_period_mean in [0.5, 1, 2, 3, 5, 8]:

    def f(x):

        df = SIRwhatever(days = 20,

                         asymptomatic_convalescent_period_mean=asymptomatic_convalescent_period_mean,

                         quiet=True,

                         daily_transmission_rate=x).run()

        v = df[df.day==10].symptomatic_velocity - 0.4

        # print("m: %f, x: %f, v: %f" % (asymptomatic_convalescent_period_mean, x, v))

        return v



    daily_transmission_rate = scipy.optimize.bisect(f, 1, 4, rtol=0.002)



    df = IsolateNDays(days = 120,

                      daily_transmission_rate = daily_transmission_rate,

                      isolation_transmission_rate_divider = 20,

                      asymptomatic_convalescent_period_mean=asymptomatic_convalescent_period_mean,

                      isolation_days=100,

                      quiet=True).run()

    print(("asymptomatic_convalescent_period_mean: %f, "+

           "daily_transmission_rate: %f, symptomatic_velocity: %f") %

         (asymptomatic_convalescent_period_mean, daily_transmission_rate, df.symptomatic_velocity[df.day == 10]))



    df.plot(x='day', y='infectious', logy=True, ax=axs[0], xlim=(15,100),

            label=('m: %f, dtr: %f' %

                   (asymptomatic_convalescent_period_mean, daily_transmission_rate)))

    df.plot(x='day', y='velocity', logy=False, ax=axs[1], xlim=(15,100),

            label=('m: %f, dtr: %f' %

                   (asymptomatic_convalescent_period_mean, daily_transmission_rate)))
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 8e4))

    axs[1].set(ylim=(-0.2, 0.45))



    for asymptomatic_ratio in [0.17, 0.3, 0.5, 0.8]:

        # we need to adjust the severe and mortality ratios accordingly

        severe_ratio = 0.05 / (1 - 0.17) * (1 - asymptomatic_ratio)

        mortality_ratio = 0.034 / (1 - 0.17) * (1 - asymptomatic_ratio)

    

        # we also need to find the daily_transmission_rate for which ever_sick_velocity is 0.4

        def f(x):

            df = SIRwhatever(days = 40,

                             asymptomatic_ratio=asymptomatic_ratio,

                             mortality_ratio=mortality_ratio,

                             severe_ratio=severe_ratio,

                             quiet=True,

                             daily_transmission_rate=x).run()

            v = df[df.day==20].observed_ever_symptomatic_velocity - 0.4

            # print("m: %f, x: %f, v: %f" % (asymptomatic_convalescent_period_mean, x, v))

            return v



        daily_transmission_rate = scipy.optimize.bisect(f, 1, 4, rtol=0.002)

        print("asymptomatic_ratio: %f, severe_ratio: %f, mortality_ratio: %f, daily_transmission_rate: %f" %

              (asymptomatic_ratio, severe_ratio, mortality_ratio, daily_transmission_rate))

    

        df = IsolateNDays(days = 150,

                          daily_transmission_rate=daily_transmission_rate,

                          isolation_transmission_rate_divider=10,

                          asymptomatic_ratio=asymptomatic_ratio,

                          mortality_ratio=mortality_ratio,

                          severe_ratio=severe_ratio,

                          isolation_days=200).run()



        label = " ar: %f" % asymptomatic_ratio

        ys = ['infectious', 'symptomatic', 'death', 'immune', 'infectable']

        df.plot(x='day', y=ys, logy=True, ax=axs[0], xlim=(15,100),

                label=[y + label for y in ys])

        velocity_ys = ['velocity', 'symptomatic_velocity']

        df.plot(x='day', y=velocity_ys, logy=False, ax=axs[1], xlim=(15,100),

                label=[y + label for y in velocity_ys])
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 7e6))

    axs[1].set(ylim=(-0.4, 0.45))



    for asymptomatic_ratio in [0.17, 0.4, 0.8]:

        # we need to adjust the severe and mortality ratios accordingly

        severe_ratio = 0.05 / (1 - 0.17) * (1 - asymptomatic_ratio)

        mortality_ratio = 0.034 / (1 - 0.17) * (1 - asymptomatic_ratio)

    

        # we also need to find the daily_transmission_rate for which ever_sick_velocity is 0.4

        def f(x):

            df = SIRwhatever(days = 40,

                             asymptomatic_ratio=asymptomatic_ratio,

                             mortality_ratio=mortality_ratio,

                             severe_ratio=severe_ratio,

                             quiet=True,

                             daily_transmission_rate=x).run()

            v = df[df.day==20].observed_ever_symptomatic_velocity - 0.4

            # print("m: %f, x: %f, v: %f" % (asymptomatic_convalescent_period_mean, x, v))

            return v



        daily_transmission_rate = scipy.optimize.bisect(f, 1, 4, rtol=0.002)

        print("asymptomatic_ratio: %f, severe_ratio: %f, mortality_ratio: %f, daily_transmission_rate: %f" %

              (asymptomatic_ratio, severe_ratio, mortality_ratio, daily_transmission_rate))

    

        df = SIRwhatever(days = 80,

                         daily_transmission_rate=daily_transmission_rate,

                         asymptomatic_ratio=asymptomatic_ratio,

                         mortality_ratio=mortality_ratio,

                         severe_ratio=severe_ratio).run()



        print("asymptomatic_ratio: %f, deaths: %f" %

             (asymptomatic_ratio, df.death.max()))

        

        label = " ar: %f" % asymptomatic_ratio

        ys = ['infectious', 'symptomatic', 'death', 'severe', 'immune', 'infectable']

        df.plot(x='day', y=ys, logy=True, ax=axs[0], xlim=(15,100),

                label=[y + label for y in ys])

        velocity_ys = ['velocity', 'symptomatic_velocity']

        df.plot(x='day', y=velocity_ys, logy=False, ax=axs[1], xlim=(15,100),

                label=[y + label for y in velocity_ys])

        
days_after = 9

observed_velocity = 0.19
# First we need to look for the daily_transmission_rate_divisor that

# makes the velocity go down to 0.19 one week after

def f(x):

    sim = IsolateNDays(days=80,

                       quiet=True,

                       isolation_transmission_rate_divider=x)

    df = sim.run()

    v = df[df.day==(sim.isolation_day + days_after)].observed_ever_symptomatic_velocity - observed_velocity

    return v



isolation_transmission_rate_divider_e2 = scipy.optimize.bisect(f, 1, 10, rtol=0.002)

print("isolation_transmission_rate_divider: %f" % isolation_transmission_rate_divider_e2)
case_e2 = IsolateNDays(days = 300,

                       isolation_transmission_rate_divider=isolation_transmission_rate_divider_e2,

                       isolation_days=500).run()
max(case_e2.death)
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 7e6))

    axs[1].set(ylim=(-0.2, 0.5))



    case_e2.plot(x='day', y=['infectious', 'symptomatic', 'death', 'immune', 'infectable'],

                 logy=True, ax=axs[0], xlim=(15,230))

    case_e2.plot(x='day', y=['velocity', 'symptomatic_velocity', 'ever_symptomatic_velocity'],

                 logy=False, ax=axs[1], xlim=(15,230))
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 7e6))

    axs[1].set(ylim=(-0.1, 0.45))



    case_e2.plot(x='day', y=['infectious', 'symptomatic', 'death', 'immune', 'infectable'],

                 logy=True, ax=axs[0], xlim=(20,80))

    case_e2.plot(x='day', y=['velocity', 'symptomatic_velocity', 'ever_symptomatic_velocity'],

                 logy=False, ax=axs[1], xlim=(20,80))
def f1(x):

    sim = SIRwhatever(days = 45,

                      quiet=True,

                      daily_transmission_rate=x)

    df = sim.run()

    v = df[df.day==(22)].observed_ever_symptomatic_velocity - 0.45

    return v

daily_transmission_rate_e2b = scipy.optimize.bisect(f1, 1, 3, rtol=0.002)

print("daily_transmission_rate: %f" % daily_transmission_rate_e2b)
def f2(x):

    sim = IsolateNDays(days = 80,

                       quiet=True,

                       daily_transmission_rate = daily_transmission_rate_e2b,

                       isolation_transmission_rate_divider=x)

    df = sim.run()

    v = df[df.day==(sim.isolation_day + days_after)].observed_ever_symptomatic_velocity - observed_velocity

    return v

isolation_transmission_rate_divider_e2b = scipy.optimize.bisect(f2, 1, 10, rtol=0.002)

print("isolation_transmission_rate_divider: %f" % isolation_transmission_rate_divider_e2b)
case_e2b = IsolateNDays(days = 300,

                        daily_transmission_rate=daily_transmission_rate_e2b,

                        isolation_transmission_rate_divider=isolation_transmission_rate_divider_e2b,

                        isolation_days=500).run()
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 7e6))

    axs[1].set(ylim=(-0.2, 0.5))



    case_e2b.plot(x='day', y=['infectious', 'symptomatic', 'death', 'immune', 'infectable'],

                  logy=True, ax=axs[0], xlim=(15,230))

    case_e2b.plot(x='day', y=['velocity', 'symptomatic_velocity', 'ever_symptomatic_velocity'],

                  logy=False, ax=axs[1], xlim=(15,230))
# we need to adjust the severe and mortality ratios accordingly

asymptomatic_ratio_e2c=0.5

severe_ratio_e2c=0.05/(1-0.17)*(1-asymptomatic_ratio_e2c)

mortality_ratio_e2c=0.034/(1-0.17)*(1-asymptomatic_ratio_e2c)



observation_delay_e2c=3

velocity0_e2c=0.45

velocity1_e2c=0.13
def f1(x):

    sim = SIRwhatever(days=45,

                      quiet=True,

                      observation_delay=observation_delay_e2c,

                      asymptomatic_ratio=asymptomatic_ratio_e2c,

                      severe_ratio=severe_ratio_e2c,

                      mortality_ratio=mortality_ratio_e2c,

                      daily_transmission_rate=x)

    df = sim.run()

    v = df[df.day==(22)].observed_ever_symptomatic_velocity

    return v - velocity0_e2c

daily_transmission_rate_e2c = scipy.optimize.bisect(f1, 1, 3, rtol=0.002)

print("daily_transmission_rate: %f" % daily_transmission_rate_e2c)
def f2(x):

    sim = IsolateNDays(days = 80,

                       quiet=True,

                       observation_delay=observation_delay_e2c,

                       asymptomatic_ratio=asymptomatic_ratio_e2c,

                       severe_ratio=severe_ratio_e2c,

                       mortality_ratio=mortality_ratio_e2c,

                       daily_transmission_rate = daily_transmission_rate_e2c,

                       isolation_transmission_rate_divider=x)

    df = sim.run()

    v = df[df.day==(sim.isolation_day + days_after)].observed_ever_symptomatic_velocity

    #print("x: %f, v: %f" % (x, v))

    return v - velocity1_e2c

isolation_transmission_rate_divider_e2c = scipy.optimize.bisect(f2, 1, 10, rtol=0.002)

print("isolation_transmission_rate_divider: %f" % isolation_transmission_rate_divider_e2c)
case_e2c = IsolateNDays(days = 400,

                        observation_delay=observation_delay_e2c,

                        asymptomatic_ratio=asymptomatic_ratio_e2c,

                        severe_ratio=severe_ratio_e2c,

                        mortality_ratio=mortality_ratio_e2c,

                        daily_transmission_rate = daily_transmission_rate_e2c,

                        isolation_transmission_rate_divider=isolation_transmission_rate_divider_e2c,

                        isolation_days=500).run()
max(case_e2c.ever_sick)
max(case_e2c.death)
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 2e6))

    axs[1].set(ylim=(-0.2, 0.5))



    case_e2c.plot(x='day', y=['observed_ever_symptomatic', 'ever_symptomatic', 'infectious',

                              'symptomatic', 'death', 'immune', 'infectable', 'severe'],

                  logy=True, ax=axs[0], xlim=(15,230))

    case_e2c.plot(x='day', y=['velocity', 'symptomatic_velocity',

                              'observed_ever_symptomatic_velocity'],

                  logy=False, ax=axs[1], xlim=(15,230))
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(20, 15))

    ax.set(ylim=(-0.2, 0.5))



    case_e2c.plot(x='day', y=['velocity', 'symptomatic_velocity',

                              'observed_ever_symptomatic_velocity'],

                  logy=False, ax=ax, xlim=(24,40))
# we need to adjust the severe and mortality ratios accordingly

asymptomatic_ratio_e2d=0.5 # info from the article

severe_ratio_e2d=0.05*0.20

mortality_ratio_e2d=0.5*0.014 # 1.4% of symptomatic patients die (ratio from the same article)



observation_delay_e2d=3

velocity0_e2d=0.40

velocity1_e2d=0.18



days_after = 9

hidden_symptomatic_ratio = 20
def f1(x):

    sim = SIRwhatever(days=45,

                      quiet=True,

                      observation_delay=observation_delay_e2d,

                      asymptomatic_ratio=asymptomatic_ratio_e2d,

                      severe_ratio=severe_ratio_e2d,

                      mortality_ratio=mortality_ratio_e2d,

                      daily_transmission_rate=x)

    df = sim.run()

    v = df[df.day==(22)].observed_ever_symptomatic_velocity

    return v - velocity0_e2d

daily_transmission_rate_e2d = scipy.optimize.bisect(f1, 1, 3, rtol=0.002)

print("daily_transmission_rate: %f" % daily_transmission_rate_e2d)
def f2(x):

    sim = IsolateNDays(days = 120,

                       quiet=True,

                       observation_delay=observation_delay_e2d,

                       asymptomatic_ratio=asymptomatic_ratio_e2d,

                       severe_ratio=severe_ratio_e2d,

                       mortality_ratio=mortality_ratio_e2d,

                       daily_transmission_rate = daily_transmission_rate_e2d,

                       ever_symptomatic_threshold = 2900 * hidden_symptomatic_ratio,

                       isolation_transmission_rate_divider=x)

    df = sim.run()

    v = df[df.day==(sim.isolation_day + days_after)].observed_ever_symptomatic_velocity

    #print("x: %f, v: %f" % (x, v))

    return v - velocity1_e2d

isolation_transmission_rate_divider_e2d = scipy.optimize.bisect(f2, 1, 10, rtol=0.002)

print("isolation_transmission_rate_divider: %f" % isolation_transmission_rate_divider_e2d)
case_e2d = IsolateNDays(days = 400,

                        observation_delay=observation_delay_e2d,

                        asymptomatic_ratio=asymptomatic_ratio_e2d,

                        severe_ratio=severe_ratio_e2d,

                        mortality_ratio=mortality_ratio_e2d,

                        ever_symptomatic_threshold = 2900 * hidden_symptomatic_ratio,

                        daily_transmission_rate = daily_transmission_rate_e2d,

                        isolation_transmission_rate_divider=isolation_transmission_rate_divider_e2d,

                        isolation_days=500).run()
max(case_e2d.ever_sick)
max(case_e2d.death)
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 7e6))

    axs[1].set(ylim=(-0.2, 0.5))



    case_e2d.plot(x='day', y=['observed_ever_symptomatic', 'ever_symptomatic', 'infectious',

                              'symptomatic', 'death', 'immune', 'infectable', 'severe'],

                  logy=True, ax=axs[0], xlim=(15,230))

    case_e2d.plot(x='day', y=['velocity', 'symptomatic_velocity',

                              'observed_ever_symptomatic_velocity'],

                  logy=False, ax=axs[1], xlim=(15,230))
# we need to adjust the severe and mortality ratios accordingly

asymptomatic_ratio_e3=0.5

severe_ratio_e3=0.05*0.20

mortality_ratio_e3=0.5*0.014

hidden_symptomatic_ratio_e3 = 1

observation_delay_e3=3

days_after_e3=11

velocity0_e3=0.35

velocity1_e3=0.15



soft_isolation_days = 14

hard_isolation_transmission_rate_divider_e3 = 8 # hopefully!
def f1(x):

    sim = SIRwhatever(days=45,

                      quiet=True,

                      observation_delay=observation_delay_e3,

                      asymptomatic_ratio=asymptomatic_ratio_e3,

                      severe_ratio=severe_ratio_e3,

                      mortality_ratio=mortality_ratio_e3,

                      daily_transmission_rate=x)

    df = sim.run()

    v = df[df.day==(22)].observed_ever_symptomatic_velocity

    return v - velocity0_e3

daily_transmission_rate_e3 = scipy.optimize.bisect(f1, 1, 3, rtol=0.002)

print("daily_transmission_rate: %f" % daily_transmission_rate_e3)
def f2(x):

    sim = IsolateNDays(days = 120,

                       quiet=True,

                       observation_delay=observation_delay_e3,

                       asymptomatic_ratio=asymptomatic_ratio_e3,

                       severe_ratio=severe_ratio_e3,

                       mortality_ratio=mortality_ratio_e3,

                       daily_transmission_rate = daily_transmission_rate_e3,

                       ever_symptomatic_threshold = 2900 * hidden_symptomatic_ratio_e3,

                       isolation_transmission_rate_divider=x)

    df = sim.run()

    v = df[df.day==(sim.isolation_day + days_after_e3)].observed_ever_symptomatic_velocity

    #print("x: %f, v: %f" % (x, v))

    return v - velocity1_e3

soft_isolation_transmission_rate_divider_e3 = scipy.optimize.bisect(f2, 1, 10, rtol=0.002)

print("isolation_transmission_rate_divider: %f" % soft_isolation_transmission_rate_divider_e3)
sim_e3 = IsolateNDays(days = 200,

                      observation_delay=observation_delay_e3,

                      asymptomatic_ratio=asymptomatic_ratio_e3,

                      severe_ratio=severe_ratio_e3,

                      mortality_ratio=mortality_ratio_e3,

                      ever_symptomatic_threshold = 2900 * hidden_symptomatic_ratio_e3,

                      daily_transmission_rate = daily_transmission_rate_e3,

                      isolation_transmission_rate_divider=soft_isolation_transmission_rate_divider_e3,

                      after_isolation_transmission_rate_divider=hard_isolation_transmission_rate_divider_e3,

                      isolation_days=14)

case_e3 = sim_e3.run()
with sns.axes_style("whitegrid"):

    fig, axs = plt.subplots(2, figsize=(20, 15))

    axs[0].set(ylim=(1, 7e6))

    axs[1].set(ylim=(-0.25, 0.45))



    case_e3.plot(x='day', y=['observed_ever_symptomatic', 'ever_symptomatic', 'infectious',

                             'symptomatic', 'death', 'immune', 'infectable', 'severe'],

                 logy=True, ax=axs[0], xlim=(15,130))

    case_e3.plot(x='day', y=['velocity', 'symptomatic_velocity',

                             'observed_ever_symptomatic_velocity'],

                 logy=False, ax=axs[1], xlim=(15,130))
sim_e3b = IsolateNDays(days = 200,

                       observation_delay=observation_delay_e3,

                       asymptomatic_ratio=asymptomatic_ratio_e3,

                       severe_ratio=severe_ratio_e3,

                       mortality_ratio=mortality_ratio_e3,

                       ever_symptomatic_threshold = 2900 * hidden_symptomatic_ratio_e3,

                       daily_transmission_rate = daily_transmission_rate_e3,

                       isolation_transmission_rate_divider=hard_isolation_transmission_rate_divider_e3,

                       isolation_days=200)

case_e3b = sim_e3b.run()
with sns.axes_style("whitegrid"):

    fig, ax = plt.subplots(figsize=(20, 15))

    ax.set(ylim=(1, 3e6), xlim=(15,130))

    ax.set_yscale('log')

    series = ['infectious', 'observed_ever_symptomatic', 'symptomatic', 'death', 'ever_sick']

    for s in series:

        sns.lineplot(data=case_e3, x ='day', y=s, ax=ax, label=('%s: soft then hard isolation' % s))

        sns.lineplot(data=case_e3b, x ='day', y=s, ax=ax, label=('%s: only hard isolation' % s))
sim_e3.daily_transmission_rate
sim_e3b.daily_transmission_rate
max(case_e3.ever_symptomatic)