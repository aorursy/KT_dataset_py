from collections import Counter

import pprint
def gotbattle2(l,d,pdict):

	if (l,d) in pdict.keys(): return (pdict[(l,d)],pdict)

	if l==0:p=0

	elif d==0:p=1

	else:

		l_wins, pdict = gotbattle2(l,d-1,pdict)

		d_wins, pdict = gotbattle2(l-1,d+1,pdict)

		p=0.5*(l_wins+d_wins)

	pdict[(l,d)]=p

	return(p,pdict)

	



def generate_gg(armysize,pdict={}):

	for size in range(armysize):

		for l in range(size+1):

			gotbattle2(l,size+1-l,pdict)

	

def extract_lists(pdict):

	llist,dlist,plist=[],[],[]

	for l,d in pdict.keys():

		p=pdict[(l,d)]

		llist.append(l)

		dlist.append(d)

		plist.append(p)

	return llist,dlist,plist
pdict={}

generate_gg(20,pdict)

pdl=list(pdict.values())

p_counts=Counter(pdl)

pprint.pprint(p_counts)
generate_gg(2000,pdict)

pdl=list(pdict.values())

p_counts=Counter(pdl)

print(p_counts[0.5])
