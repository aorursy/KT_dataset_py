%pylab inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



h = 0.95 #we want at least this chance of finding a single infected person



def samples_needed(h,p):

    n = ceil(log(1-h) / log(1-p))

    print("%s tests needed for a %s%% chance to find at least one positive, if the true prevalence is %s" %(n, h*100,p))

    return n
print("New York State, assuming 10x more cases than reported")

p = 3066.2E-6 #confirmed cases per capita in new york state. source: http://www.91-divoc.com, 29 March 2020

p*=10

__ = samples_needed(h,p)
print("nationwide average, assuming 10x reported cases exist")

p = 427E-6 #confirmed cases per capita in US

p *=10 # assume 10x underreporting



__ = samples_needed(h,p)

print("Minnesota, assuming 10x more cases than reported")

p = 89.2E-6

p*=10



__ = samples_needed(h,p)