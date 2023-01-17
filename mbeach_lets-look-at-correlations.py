#Ignore the seaborn warnings.
import theono
df = pd.read_csv('../input/primary_results.csv')
NH = df[df.state == 'New Hampshire']
g = sns.FacetGrid(NH[NH.party == 'Democrat'], col = 'candidate', col_wrap = 5)
g.map(sns.barplot, 'county', 'fraction_votes');
g = sns.FacetGrid(NH[NH.party == 'Republican'], col = 'candidate', col_wrap = 4)
g.map(sns.barplot, 'county', 'votes');
