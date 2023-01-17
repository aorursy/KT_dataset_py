import pandas as pd
import numpy as np
!pip install iminuit
import sncosmo
import pandas as pd
import time
from astropy.table import Table
from astropy import wcs, units as u
from sncosmo.bandpasses import read_bandpass
from contextlib import contextmanager
%matplotlib inline

@contextmanager
def timer(name):
    s = time.time()
    yield
    
    print('[{}] {}'.format(time.time() - s, name))

with timer('load data'):
    lc = pd.read_csv('../input/training_set.csv', nrows=10000)
    meta = pd.read_csv('../input/training_set_metadata.csv')
    meta.set_index('object_id', inplace=True)

# only use data with signal-to-noise ratio (flux / flux_err) greater than this value
minsnr = 3
# template to use
model_type = 'salt2-extended'
model = sncosmo.Model(source=model_type)


passbands = ['lsstu','lsstg','lsstr','lssti','lsstz','lssty']
with timer('prep'):
    lc['band'] = lc['passband'].apply(lambda x: passbands[x])
    lc['zpsys'] = 'ab'
    lc['zp'] = 25.0
    
object_id = 1598

data = Table.from_pandas(lc[lc.object_id == object_id])

photoz = meta.loc[object_id, 'hostgal_photoz']
photoz_err = meta.loc[object_id, 'hostgal_photoz_err']

# run the fit
with timer('fit_lc'):
    result, fitted_model = sncosmo.fit_lc(
        data, model,
        model.param_names,
        # sometimes constant bound ('z':(0,1.4)) gives better result, so trying both seems better
        bounds={'z':(max(1e-8,photoz-photoz_err), photoz+photoz_err)},
        minsnr=minsnr)  # bounds on parameters

sncosmo.plot_lc(data, model=fitted_model, errors=result.errors, xfigsize=10)


print('chisq:{}'.format(result.chisq))
print('hostgal_photoz: {}, hostgal_specz: {}, estimated z by the model: {}'.format(meta.loc[object_id,'hostgal_photoz'],
                                                                                   meta.loc[object_id,'hostgal_specz'],
                                                                                   result.parameters[0]))
df = pd.DataFrame(columns=['chisq'] + model.param_names)
df.index.name = 'object_id'
df.loc[object_id] = [result.chisq] + list(result.parameters)
df