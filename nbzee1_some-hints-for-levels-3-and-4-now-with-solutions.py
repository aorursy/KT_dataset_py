import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import random

import math

import string

import itertools

import copy



from fuzzywuzzy import fuzz 

from tqdm import tqdm
# After figuring out / forgetting / re-figuring out the padding convention a few times, I just wrote a function to automate it.

def padding(s):

	if len(s)%100==0:

		return s

	else:

		pad = 100 - len(s)%100

		parity = pad%2

		offset = pad//2

		if parity == 1:

			return ("*"*(offset))+s+("*"*(offset+1))

		else:

			return ("*"*(offset))+s+("*"*offset)



# Siimlarly, we need to be able to cut a padded text back down to size in a consistent way.

crop = lambda s,l: s[len(s)//2-sum(divmod(l,2)):len(s)//2+l//2]



# Character frequency analysis on an iterable of strings (list, series, etc).

def char_freq(series):

	chars = {i:0 for i in string.ascii_letters + string.digits + string.punctuation}

	for s in series:

		for i in s:

			try:

				chars[i]+=1

			except:

				chars[i]=1

	return pd.Series(chars).sort_values(ascending=False)



# Encryption and decryption functions for levels 1 and 2. Level 2 is an example of just how weird solutions can look if you're

# not thinking about them quite right: I got thinking in terms of rectangles early on in the solving process and never quite

# recovered.



def decrypt_1(c,s=15):

	pattern = [15,24,11,4]

	res = ""

	pos = pattern.index(s)

	for i in range(len(c)):

		if c[i].isalpha() and c[i].lower()!="z":

			n = ord(c[i]) - pattern[pos]

			if (c[i].islower() and n<97) or (c[i].isupper() and n<65):

				n+=25

			res+=chr(n)

			pos = (pos+1)%4

		else:

			res+=c[i]

	return res



def encrypt_1(c,s=15):

	pattern = [15,24,11,4]

	res = ""

	pos = pattern.index(s)

	for i in range(len(c)):

		if c[i].isalpha() and c[i].lower()!="z":

			n = ord(c[i]) + pattern[pos]

			if (c[i].islower() and n>121) or (c[i].isupper() and n>89):

				n+=-25

			res+=chr(n)

			pos = (pos+1)%4

		else:

			res+=c[i]

	return res



def encrypt_2(s):

	blocks = []

	left,right= math.ceil(len(s)/40),len(s)//40

	for k in range(0,right):

		blocks+=[[i for i in s[40*k:40*k+21]]]

		blocks+=[[np.nan]+[i for i in s[40*(k+1)-1:k*40+20:-1]]+[np.nan]]

	if left-right==1:

		blocks+=[[i for i in s[40*right:]]+[np.nan]]

	return "".join(blocks[i][j] for j in range(len(blocks[0])) for i in range(len(blocks)) if not pd.isna(blocks[i][j]))



def decrypt_2(s):

	top,bottom= math.ceil(len(s)/40),len(s)//40

	blocks = [s[:top]]+[s[top + k*(top+bottom):top + (k+1)*(top+bottom)] for k in range(0,19)]+[s[-bottom:]]

	m =  "".join(["".join([blocks[0][k]]+[blocks[i][2*k] for i in range(1,20)]+[blocks[20][k]]+[blocks[i][2*k+1] for i in range(19,0,-1)]) for k in range(bottom)])

	if top - bottom ==1:

		m+="".join(blocks[k][-1] for k in range(20))

	return m



train_df = pd.read_csv('../input/ciphertext-challenge-iii/train.csv')

train_df.set_index("index",inplace=True)



# Drop all plaintexts that have already been assigned in levels 1 and 2.

assigned = pd.read_csv('../input/submission2a/submission2a.csv').rename(columns={"index":"p_index1"})

train = train_df.drop(index=list(assigned[assigned.p_index1!=-1].p_index1))





# Set up the basics: text length, padded text, and padded text length.

train['length'] = train['text'].str.len()

train['padded_text'] = train.apply(lambda x: padding(x.text),axis = 1)

train['padded_length']=train['padded_text'].str.len()



test_data = pd.read_csv('../input/ciphertext-challenge-iii/test.csv')

test_data["length"] = test_data.ciphertext.str.len()

test_3 = test_data[test_data.difficulty==3].copy()

test_3["unsplit_length"] = test_3.ciphertext.str.len()

test_3['length'] = test_3.apply(lambda x: len(x.ciphertext.split()),axis=1)



# Pair up a single example on length grounds and compute the four possible versions of its level 2 encryption.

c1 = test_3.loc[60920,"ciphertext"]

p1 = train.loc[34509,"padded_text"]

p1e = [encrypt_2(encrypt_1(p1,i)) for i in [15,24,11,4]]
# Find some numbers in c1 which appear more than once

c1_numbers = pd.Series(c1.split()).value_counts()

print("Most common numbers in our chosen ciphertext example: \n{}".format(c1_numbers.sort_values(ascending=False).head()))
# Pick out the indices where those repeated numbers appear

c1_repetitions = [(j,i) for j in c1_numbers[c1_numbers>1].index.tolist()[:5] for i in range(len(c1.split())) if c1.split()[i]==j]



# Have a look at what those numbers map to in each of the four lv2-encoded plaintexts

pd.DataFrame({(j,i):[p1e[k][i] for k in range(4)] for (j,i) in c1_repetitions},index=['encrypt15','encrypt24','encrypt11','encrypt04'])
# Store the information about that first pair 

test_3.loc[60920,"p_index"] = 34509

test_3.loc[60920,"lvl1_key"] = 4



# Get a list of all the numbers that appear in all of the ciphertexts

combined_numbers = pd.Series([int(i) for i in list(itertools.chain.from_iterable(test_3.ciphertext.str.split()))])



print("Min: {}, Max: {}, Count of distinct numbers: {}.\n\n".format(combined_numbers.min(), combined_numbers.max(),len(set(combined_numbers))))



print("Most common numbers and their frequencies: ")

displaydf = pd.DataFrame([combined_numbers.value_counts().sort_values(ascending=False).head(10),combined_numbers.value_counts().sort_values(ascending=False).head(10)/len(combined_numbers)]).T

displaydf.rename(columns={0:'count',1:'percent of total'}).style.format({'count': "{:.0f}",'percent of total': "{:.2%}"})
# Function to update the decryption dictionary

# Note that the dict values are sets, which is mildly irritating when we want to actually decrypt something later, but it's

# great for catching errors: if we accidentally mismatch a ciphertext/plaintext pair, we'll end up assigning multiple 

# characters to one number and the function runs a check at the end to catch this and revert to the last safe dictionary.



def update_dict(d_dict,test_df):

    current_len = len(d_dict)

    current_dict = copy.deepcopy(d_dict) # in case we screw up

    df = test_df[~pd.isna(test_df.p_index)]

    for i,c in tqdm(df.iterrows()):

        ptext = encrypt_2(encrypt_1(train.loc[c.p_index,"padded_text"],c.lvl1_key))

        csplit = c.ciphertext.split()

        for j in range(len(csplit)):

            if ptext[j]!="*":

                try:

                    d_dict[csplit[j]].add(ptext[j])

                except:

                    d_dict[csplit[j]] = set([ptext[j]])

    # now verify that we haven't screwed up:

    mult_assign = [i for i in d_dict.keys() if len(d_dict[i])>1] # check if we've assigned two characters to one number

    long_str_assign = [i for i in d_dict.keys() if len(list(d_dict[i])[0])>1] # check if we've assigned a longer string to a number

    if mult_assign or long_str_assign:

        print("Something's gone wrong... Revert? Y/n")

        reply = input(" ")

        # default behaviour here is to revert to the previous dict: if you didn't mean to, just run the function again

        if reply not in ["N","n"]:

            d_dict = copy.deepcopy(current_dict)       

    else:

        print("Dictionary update success! We have {} new entries, for a total of {}".format(len(d_dict)-current_len,len(d_dict)))

        print("Updating the translations for the test dataset...")

        for i,c in tqdm(test_3.iterrows()):

            test_3.loc[i,"partial_translate"]=translate_3(c.ciphertext,d_dict)

        print("Done!")





# Decrypts a ciphertext from level 3 to 2, as much as is possible; missing chars are denoted by "*"

def translate_3(s,d_dict):

    res = ""

    for i in s.split():

        if i in d_dict.keys():

            res+=list(d_dict[i])[0]

        else:

            res+="*"

    return res





# Bruteforce search for ciphertext/plaintext matches based on partial decryption

# You can restrict which subset of level 3 ciphertexts to consider (df_to_test) and which subset of the plaintexts (df_train)

# to compare them to. 

# Ciphertexts with fewer than min_length known characters are skipped, and potential matches with more than fp_threshold uncertainties

# (i.e. matches to plaintext padding characters) are dropped.

def potential_match_search(df_to_test,df_train,d_dict,min_length,fp_threshold):

    print("Searching for matches for {} ciphertexts...".format(len(df_to_test)))

    res = {}

    for a,c in tqdm(df_to_test.iterrows()):

        # Pick out the indices in the ciphertext whose decryption is known

        c_pos = [s for s in range(len(c.partial_translate)) if c.partial_translate[s]!="*"]

        c_trunc = "".join([c.partial_translate[s] for s in c_pos])

        # user-set threshold requires a minimum number of known points to proceed

        if len(c_trunc)>=min_length:

            possibles = []

            for i,p in df_train[df_train.padded_length==c.length].iterrows():

                for l in ["encrypt15","encrypt24","encrypt11","encrypt04"]:

                    t = "".join([df_train.loc[i,l][j] for j in c_pos])

                    # If the corr. character in the plaintext is known, we require it to match the ciphertext.

                    # We also allow the plaintext char to be unknown (i.e. comes from the padding)

                    if (sum(t[k] in [c_trunc[k],"*"] for k in range(len(c_trunc)))>=min_length) and (t.count("*") < fp_threshold):

                        possibles+=[(i,t,int(l[-2:]))]

            res[a] = possibles[:]

    return res

  

    
test_3["p_index"] = np.nan

test_3["lvl1_key"] = np.nan



# These examples were matched by hand.

for (i,j,k) in zip([60920,99421,75719,746, 2734, 10978, 30192, 70167],[34509,31644,76893,93461, 40234, 47443, 77656,76309],[4,4,11,11,24,15,11,24]):

	test_3.loc[i,"p_index"] = j

	test_3.loc[i,"lvl1_key"] = k





print("Performing setup... (takes a while but only has to be done once)")

train["encrypt15"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,15)),axis=1)

train["encrypt24"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,24)),axis=1)

train["encrypt11"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,11)),axis=1)

train["encrypt04"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,4)),axis=1)    



decrypt_3_dict = {}

update_dict(decrypt_3_dict,test_3)

# Start with the low-hanging fruit: texts of length >=200. 

result = potential_match_search(test_3[test_3.length==200],train[train.padded_length==200],decrypt_3_dict,min_length = 30,fp_threshold = 30)

exacts = [i for i in result.keys() if len(result[i])==1]

print("Exact matches: {} of {}.".format(len(exacts),len(test_3[test_3.length==200])))



for i in exacts:

    test_3.loc[i,"p_index"] = result[i][0][0]

    test_3.loc[i,"lvl1_key"] = int(result[i][0][2])



update_dict(decrypt_3_dict,test_3)
result = potential_match_search(test_3[(test_3.length==200)&(pd.isna(test_3.p_index))],train[(train.padded_length==200)&(train.length>=90)],decrypt_3_dict,min_length = 40,fp_threshold = 40)

exacts = [i for i in result.keys() if len(result[i])==1]

print("Exact matches: {} of {}.".format(len(exacts),len(test_3[(test_3.length==200)&(pd.isna(test_3.p_index))])))



for i in exacts:

    test_3.loc[i,"p_index"] = result[i][0][0]

    test_3.loc[i,"lvl1_key"] = int(result[i][0][2])



update_dict(decrypt_3_dict,test_3)
print("Most common numbers and their mappings:")

for i in list(pd.Series(combined_numbers).value_counts().sort_values(ascending=False).index)[:20]:

    if str(i) in decrypt_3_dict.keys():

        print("{} -> {}".format(i,list(decrypt_3_dict[str(i)])[0]))
decryption_image=pd.Series([list(decrypt_3_dict[i])[0].lower() for i in decrypt_3_dict.keys()]).value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(15, 8))

ax.bar(decryption_image.index[0:30],decryption_image[0:30],label="Ciphertext")

plt.ylabel('Character count',fontsize=16)

plt.title('Frequency of the top 30 characters in our decryption dictionary',fontsize=16)

plt.show()
print("".join(list(decrypt_3_dict[str(i)])[0] if str(i) in decrypt_3_dict.keys() else "*" for i in range(1500)))
# level3key.txt is https://www.gutenberg.org/files/46464/46464-0.txt

with open('../input/level3key/level3key.txt',"r") as f:

	key = f.readlines()



key[0] = ' '+key[0]

splitkey = "".join([i for j in key for i in j])

splitkey = splitkey.replace('\n','  ')

splitkey = {i:splitkey[i] for i in range(len(splitkey))}



def decrypt_3(s,translate_3 = splitkey):

	res = ""

	for i in s.split():

		if int(i) in translate_3.keys():

			res+=list(translate_3[int(i)])[0]

		else:

			res+="*"

	return res

c1 = test_3.iloc[0,1]

print("Test it out on an example:\n\n{} \n\ndecrypts to \n\n{}".format(c1,decrypt_1(decrypt_2(decrypt_3(c1)))))
test_4 = test_data[test_data.difficulty==4].copy()

test_4.loc[1,"ciphertext"]
mean3 =(test_3.unsplit_length/test_3.length).mean()

print("Average number of characters in a level 3 string per character of decrypted text: {0:.2f}\n".format(mean3))

print("Lengths of ciphertext strings after we undo the base64 encoding: \n{}\n".format(round(test_4.length*(3/4),-1).value_counts()))

print("Lengths of ciphertext strings after we also divide by the level 3 mean: \n{}".format(round(test_4.length*(3/4)/mean3,-1).value_counts()))
test_4['lvl4_length'] = test_4.ciphertext.str.len()

test_4['length'] = test_4.apply(lambda x: int(round(x.lvl4_length/7.75,-2)),axis=1)
lvl4_firstchars = pd.Series([x.ciphertext[:2] for i,x in test_4.iterrows()])

print("All first two character combos from level 4 ciphertexts and their frequencies (% of examples): \n{}".format(lvl4_firstchars.value_counts().sort_values(ascending=False)/len(test_4)*100))
standard_b64chars = string.ascii_uppercase+string.ascii_lowercase+string.digits+"+/"

b64_to_num = {j:i for (i,j) in zip(range(64),standard_b64chars)}

num_to_b64 = {i:j for (i,j) in zip(range(64),standard_b64chars)}

to_binary = lambda x: "{0:06b}".format(x)



conversions = {s:int('0b'+(to_binary(b64_to_num[s[0]])+to_binary(b64_to_num[s[1]]))[:8],2) for s in list(lvl4_firstchars.value_counts().index)}



print("First decoded character of each level 4 string:")

for s in list(lvl4_firstchars.value_counts().index):

    print("{} -> {}".format(s,conversions[s]))
def decrypt_base64(s,key1=b64_to_num):

	if len(s)%4!=0:

		s1 = s+"="*(len(s)%4)

	else:

		s1 = s[:]

	blocks = len(s1)//4 # now always guaranteed to divide neatly

	pad = s1[-3:].count("=")

	s1 = s1.translate({ord("="): None})

	converted = "".join(['{0:06b}'.format(key1[i]) for i in s1])

	converted = ["0b"+converted[8*i:8*(i+1)] for i in range(blocks*3)]

	if pad >0:

		converted = converted[:-pad]

	return [int(i,2) for i in converted]



test_4["base64"] = test_4.apply(lambda x: decrypt_base64(x.ciphertext),axis=1)

print("Have a look at the first few characters of some random examples...")

for i in range(10):

    print(test_4.iloc[i,-1][:20])
print("Frequency of each number in the first position of the base64 decoded level 4 ciphertexts (% of examples) :\n{}".format(pd.Series([x.base64[0] for i,x in test_4.iterrows()]).value_counts().sort_values(ascending=False)/len(test_4)*100))

print("Frequency of each digit in the first character of level 3 ciphertexts (% of examples) :\n{}".format(pd.Series([x.ciphertext[0] for i,x in test_3.iterrows()]).value_counts().sort_values(ascending=False)/len(test_3)*100))
# first, get a list of the numbers associated to each character position

# we use a set for now (simplifies dealing with duplicates; we'll convert to a list later)

# every example has length >= 550 so let's look at those first:

alphabet=[]

for j in range(550):

	alphabet+= [set([x.base64[j] for i,x in (test_4.iloc[:500]).iterrows()])]



# see where we didn't get enough data:

missing = [i for i in range(len(alphabet)) if len(alphabet[i])<11]



# rerun on the entire dataset to fill in the blanks

for j in missing:

	alphabet[j] = set([x.base64[j] for i,x in test_4.iterrows()])



# listify and sort

alphabet = [sorted(list(j)) for j in alphabet]



    

# now pick out the whitespace value for each character position 

# In hindsight there's probably a better way to do this, but at the time I didn't really know where I was going with this



# Whitespace is always separated from the other numbers, but sometimes 8,9 are a separate block too

# Luckily, there are only 3! ways to arrange the blocks [whitespace][numbers corr to 1 - 7][numbers corr to 8, 9]

# So associate a binary number to each pattern to identify it, convert it to an integer, then store where the whitespace is for that pattern

pattern_mapping = {514:0,640:0,6:-3,5:-1,257:-1,384: 2}





# build a dictionary that just lists what number maps to whitespace at each step

whitespace_dict = {}



# extract the whitespace character for each position

# we skip 0 for now because position 0 is never whitespace

for i in range(1,len(alphabet)):

	a = alphabet[i]

	gap_pattern = int("0b"+"".join(["0" if a[i]-a[i-1]==1 else "1" for i in range(1,11)]),2)

	if gap_pattern in pattern_mapping.keys():

		whitespace_dict[i] = a[pattern_mapping[gap_pattern]]

	elif a[1] - a[0]!=1: # so we can separate the whitespace but not the other

		whitespace_dict[i] = a[0]

	elif a[-1] - a[-2]!=1: # as previous

		whitespace_dict[i] = a[-1]



print("Truncated to first 30 values for display purposes \n\n Each line contains:\nwhitespace_value: [the numbers that decrypt to 0, 1,...,9 when whitespace decrypts to whitespace_value] \n \n")

# now print out the results (truncated to first 30 for display purposes)

for i in range(30):

	if [k for k in whitespace_dict.keys() if whitespace_dict[k]==i]!=[]:

		print(i,[j for j in alphabet[[k for k in whitespace_dict.keys() if whitespace_dict[k]==i][0]] if j!=i])

	else:

		print(i)

test_4["b64_length"] = test_4.apply(lambda x:len(x.base64),axis=1)



# XORing the ciphertext and the plaintext together would generate the key, but we don't know the plaintexts yet

# However, we do know that the plaintext is one of 11 characters.

# So XOR the ciphertext character in position i with all 11 possible plaintext characters: the key must be one of these 11 numbers.

# Repeat for all ciphertexts and take the intersection: that's the key value for position i.



key_bags = {}

vals = [ord(i) for i in string.digits+" "]



# Rather than doing this to the whole test set, we'll start with just the longer texts and see if that gets us enough information

for j in tqdm(range(1152)):

	res = set(range(256))

	for i,c in test_4[test_4.length>=200].iterrows():

		if c.b64_length>j:

			res &= set([c.base64[j]^k for k in vals])

	key_bags[j]=res



missing = [i for i in key_bags.keys() if len(key_bags[i])!=1]



# There's a few that didn't quite get narrowed down, so we throw the rest of the test set at those indices only:

for j in tqdm(missing):

	res = set(range(256))

	for i,c in test_4.iterrows():

		if c.b64_length>j:

			res &= set([c.base64[j]^k for k in vals])

	key_bags[j]=res



# We can figure out 0 by hand.

key_bags[0] = set([49])



# finally, collect the results together in a list.

key_list = [list(key_bags[i])[0] for i in range(1152)]
def decrypt_4(s):

	tob64 = decrypt_base64(s)

	return "".join([chr(key_list[i]^tob64[i]) for i in range(min(len(key_list),len(tob64)))])

c1 = test_4.iloc[0,1]

print("Test it out on an example:\n\n{} \n\ndecrypts to \n\n{}".format(c1,decrypt_1(decrypt_2(decrypt_3(decrypt_4(c1))))))