print ("hllo work")

%matplotlib inline
counts = train.groupby(['Dates', 'X', 'Y']).size()

counts.value_counts().plot('bar', logy=True)
counts.value_counts()
other = pd.DataFrame(counts[counts>=13])
other = other.reset_index()
manyarrests = train.merge(other, how='right')
manyarrests[manyarrests[0]==16]
manyarrests[manyarrests[0]==14]