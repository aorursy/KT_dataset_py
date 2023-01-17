import pandas as pd
import seaborn as sns
import pylab as plt
df = pd.read_csv('../input/mriqc-data-cleaning/t1w.csv')
filtered_df = df[df['bids_meta.ManufacturersModelName'].isin(['Prisma', 'Tim Trio'])]
len(filtered_df)
filtered_df['bids_meta.ManufacturersModelName'].value_counts().plot.pie()
filtered_df[filtered_df['bids_meta.ManufacturersModelName'] == 'Prisma'].groupby(['size_x', 'size_y', 'size_z', 'spacing_x', 'spacing_y', 'spacing_z']).count()['provenance.md5sum'].sort_values(ascending=False).head(20)
filtered_df[filtered_df['bids_meta.ManufacturersModelName'] == 'Trio'].groupby(['size_x', 'size_y', 'size_z', 'spacing_x', 'spacing_y', 'spacing_z']).count()['md5sum'].sort_values(ascending=False).head(20)
specific_protocol_df = filtered_df[(filtered_df.size_x == 176) & (filtered_df.size_y == 256) & (filtered_df.size_z == 256) & (filtered_df.spacing_x == 1.0) & (filtered_df.spacing_y == 1.0) & (filtered_df.spacing_z == 1.0)]
print(specific_protocol_df.groupby(['size_x', 'size_y', 'size_z', 'spacing_x', 'spacing_y', 'spacing_z']).count()['provenance.md5sum'].sort_values())
specific_protocol_df['bids_meta.ManufacturersModelName'].value_counts().plot.pie()
sns.factorplot(x="bids_meta.ManufacturersModelName", y="snr_wm", data=specific_protocol_df, kind='violin')
sns.factorplot(x="bids_meta.ManufacturersModelName", y="snr_gm", data=specific_protocol_df, kind='violin')
sns.factorplot(x="bids_meta.ManufacturersModelName", y="cnr", data=specific_protocol_df, kind='violin')
sns.factorplot(x="bids_meta.ManufacturersModelName", y="cjv", data=specific_protocol_df, kind='violin')