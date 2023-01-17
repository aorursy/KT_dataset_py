!pip install vaex==2.5.0
import vaex as vx
# vx.datasets.helmi_de_zeeuw_10percent.url = 'https://github.com/vaexio/vaex-datasets/raw/master/datasets/helmi-dezeeuw-2000-10p.hdf5'
dfv = vx.open("../input/vaex-example-dataset/helmi-dezeeuw-2000-10p.hdf5")
dfv
dfv['indx'] = vx.vrange(1, len(dfv)+1)
dfv['indx']
dfv['seg_id'] = dfv.apply(lambda x: x/1000 + 10001, arguments=[dfv['indx']] )
dfv['seg_id']
