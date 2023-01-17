import pandas as pd



df_ks = pd.read_csv('../input/blackmores-product-on-ecommerce-in-indonesia/Data Blackmores Kalbestore.csv')

df_tp = pd.read_csv('../input/blackmores-product-on-ecommerce-in-indonesia/BlackmoresTokopedia.csv')
df_tp.head()

df_bio_ace = df_tp.loc[df_tp['keyword'] == 'BLACKMORES BIO ACE EXCELL (30)']
df_data = df_bio_ace[['price_int','keyword']]
df_data
df_data.plot.box(grid='True')