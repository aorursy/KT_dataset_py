!pip install git+https://github.com/brazil-data-cube/wlts.py
import wlts
service = wlts.WLTS('http://brazildatacube.dpi.inpe.br/dev/wlts')
service
service.collections
service['mapbiomas_amz_4_1']
# service['mapbiomas_fabi']
tj = service.tj(latitude=-12.0, longitude=-54.0, collections='mapbiomas_amz_4_1,terraclass_amz')
tj
tj.df()
tj2 = service.tj(latitude=-3.014626, longitude=-55.884531, collections='mapbiomas_amz_4_1,prodes_amz_legal,terraclass_amz')
tj2.df()
tj3 = service.tj(latitude=-3.558805, longitude=-53.305502, collections='mapbiomas_amz_4_1,prodes_amz_legal')
tj3.df()
tj_null = service.tj(latitude=-23.238155, longitude=-45.018637, collections='mapbiomas_amz_4_1')
tj_null
!pip install git+https://github.com/brazil-data-cube/sample.py
!pip install wtss
import sample
from wtss import *
from matplotlib import pyplot as plt
sample_service = sample.sample(wfs="http://brazildatacube.dpi.inpe.br/bdc/geoserver", auth=("reader", "G30r3@d3rGS"))
sample_service.datasets
sample_service.dataset('BDC Sample Dataset - Bahia Test Area')
obs_bdc = sample_service.dataset('BDC Sample Dataset - Bahia Test Area').observation
fig, ax = plt.subplots(figsize=(20,15))

obs_bdc.plot(ax=ax, cmap='Set2', column='class_name',edgecolor='black', legend=True,legend_kwds={'title': "Classes", 'fontsize': 15})
obs_bdc.loc[obs_bdc['class_name'] == 'Pasture'].head()
obs_bdc.to_file("my_save_bdc_obs.shp",  encoding="utf-8")
tj_bdc = service.tj(latitude=obs_bdc.iloc[93]['location'].y , longitude=obs_bdc.iloc[93]['location'].x, collections='mapbiomas_cerrado_4_1,prodes_cerrado')
tj_bdc.df()
service = WTSS('http://www.esensing.dpi.inpe.br')
coverage = service['MOD13Q1']
ts = coverage.ts(attributes=('red', 'nir', 'ndvi'),
                 latitude=-12.0, longitude=-54.0,
                 start_date='2018-09-01', end_date='2019-08-31')
ts.plot()