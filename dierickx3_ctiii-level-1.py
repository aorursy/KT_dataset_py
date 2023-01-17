# MODULES TO IMPORT

import math

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import re

import seaborn as sns
# CONFIGURE PANDAS OUTPUT

pd.set_option('display.max_columns', 999)

pd.set_option('display.width', 9999)

pd.set_option('display.max_colwidth', 9999)

pd.set_option('display.html.table_schema', True)



# CUSTOM DATA FRAME STYLE CAN APPLY BY CALLING

# df.head().style.set_table_styles(tom_df_style)

tom_df_style = [

    dict(selector="td", props=[('font-family','Consolas'), ('font-size','8pt'), ('vertical-align','top'), ('text-align','left'), ('white-space', 'pre')]),

    dict(selector="th", props=[('font-family','Consolas'), ('font-size','8pt'), ('vertical-align','top'), ('text-align','left'), ('white-space', 'pre')])

]
# LOAD IN RAW (T)EXT DATASET AND ADD TEXT LENGTH VARIABLE

dft = pd.read_csv("/kaggle/input/train.csv")

dft["text_length"] = dft["text"].str.len()

dft.head(7).style.set_table_styles(tom_df_style)
# MOST ARE <100 CHARACTERS

sns.distplot(dft["text_length"], kde=False, rug=True);
# LOAD IN (E)NCRYPTED DATASET AND ADD TEXT LENGTH VARIABLE

dfe = pd.read_csv("/kaggle/input/test.csv")

dfe["ciphertext_length"] = dfe["ciphertext"].str.len()

dfe.head(7).style.set_table_styles(tom_df_style)
# BREAK OUT THE ENCRYPTED DATA SET INTO SEPARATE ONES BASED ON DIFFICULTY LEVEL, SINCE IT SOUNDS LIKE THEY'RE APPLIED IN SUCCESSION (1 -> 2 -> 3 ->4), SO LET'S JUST WORRY ABOUT TRYING TO "CRACK" LEVEL 1 FIRST

dfe1 = dfe[dfe["difficulty"] == 1].copy()

dfe2 = dfe[dfe["difficulty"] == 2].copy()

dfe3 = dfe[dfe["difficulty"] == 3].copy()

dfe4 = dfe[dfe["difficulty"] == 4].copy()
# LEVEL 1 APPEARS TO "JUST" BE SCRAMBLED IN SOME WAY

dfe1.sort_values(["ciphertext_length","ciphertext"], ascending=[True, True]).head(10).style.set_table_styles(tom_df_style)
# LEVEL 2 APPEARS TO MAYBE BE SCRAMBLED AGAIN

dfe2.sort_values(["ciphertext_length","ciphertext"], ascending=[True, True]).head(7).style.set_table_styles(tom_df_style)
# LEVEL 3 FURTHER ENCRYPTED INTO SOME NUMERIC CODING

dfe3.sort_values(["ciphertext_length","ciphertext"], ascending=[True, True]).head(7).style.set_table_styles(tom_df_style)
# LEVEL 4 LOOKS LIKE FURTHER MORE ADVANCED BYTE ENCODING

dfe4.sort_values(["ciphertext_length","ciphertext"], ascending=[True, True]).head(7).style.set_table_styles(tom_df_style)
# WHAT'S THE DISTRUBITION OF THE ENCRYPTED STRING LENGTHS?

dfe1.groupby(["ciphertext_length"]).agg("count")
# FIND THE TWO LEVEL 1 ENCRYPTED PHRASES 400+ CHARACTERS WIDE

dfe1[dfe1["ciphertext_length"] >= 400].sort_values(["ciphertext"]).style.set_table_styles(tom_df_style)
# FIND WHAT INPUT PHRASES ARE AT LEAST 400+ CHARACTERS WIDE; WE KNOW THE ABOVE TWO ENCRYPTED PHRASES *MUST* CORRELATE TO ONE OF THESE INPUT PHRASES (NOTE: THERE ARE THREE OF THEM)

dft[(dft["text_length"] >= 400) & (dft["text_length"] <= 500)].sort_values(["text"]).style.set_table_styles(tom_df_style)
# LET'S TAKE AN EVEN CLOSER LOOK HERE

example_encrypted_text = dfe1[dfe1["ciphertext_id"]=="ID_6100247c5"]["ciphertext"].values[0]

example_plain_text     = dft[dft["plaintext_id"]=="ID_f000cad17"]["text"].values[0]



# HAVE TO "PAD" THIS PLAIN TEXT MESSAGE WITH 6 SPACES TO ACCOUNT FOR FINDING #6 ABOVE

example_plain_text = "      " + example_plain_text

pd.DataFrame([example_plain_text, example_encrypted_text], columns=["Text"], index=["ptext", "etext"]).style.set_table_styles(tom_df_style)
# WHAT OTHER ENCRYPTED MESSAGES CONTAIN "YSHEAPA" (AKA, "NORFOLK" ?)

dfe1[(dfe1["ciphertext"].str.contains("YSHEAPA"))].sort_values(["ciphertext"]).style.set_table_styles(tom_df_style)
# WE SEE A "YSHEAPA: Ssi" ABOVE, IS THERE A "NORFOLK: The " HERE? YEP! 

dft[dft["text"].str.contains("NORFOLK: The ")].sort_values(["text"]).style.set_table_styles(tom_df_style)
# WITH THE HELP OF THE APOSTROPHE SEEN (WHICH WE KNOW ARE "PRESERVED") IN "The Cardinal's" 

# WE CAN PROBABLY GUESS THESE GO TOGETHER (ALTHOUGH: SHIFTED 27 SPACES INSTEAD OF 6..HMMM)

example_encrypted_text = dfe1[dfe1["ciphertext_id"]=="ID_9bf75d21c"]["ciphertext"].values[0]

example_plain_text     = dft[dft["plaintext_id"]=="ID_9b8e655fe"]["text"].values[0]

example_plain_text = (" " * 27) + example_plain_text

pd.DataFrame([example_plain_text, example_encrypted_text], columns=["Text"], index=["ptext", "etext"]).style.set_table_styles(tom_df_style)
# IT WOULD APPEAR THAT THE PLAIN TEXT MESSAGES ARE FIRST "CENTERED" BEFORE BEING ENCRYTED

list_text = list(dft["text"])

list_lenx = [int(math.ceil(x / 100.0)) * 100 for x in dft["text_length"]]



dft["text_adj"] = list(map(lambda t, x: t.center(x,"`"), list_text, list_lenx))

dft.head().style.set_table_styles(tom_df_style)
# LETS TRY THIS AGAIN

example_encrypted_text = dfe1[dfe1["ciphertext_id"]=="ID_9bf75d21c"]["ciphertext"].values[0]

example_plain_text     = dft[dft["plaintext_id"]=="ID_9b8e655fe"]["text_adj"].values[0]

pd.DataFrame([example_plain_text, example_encrypted_text], columns=["Text"], index=["ptext", "etext"]).style.set_table_styles(tom_df_style)
# PERHAPS ADJUSTED SENTENCE PATTERNS ARE "UNIQUE" AND CAN PROVIDE US WITH A UNIQUE "SENTENCE SIGNATURE" (SINCE DON'T YET HAVE A FULL CHARACTER-FOR-CHARACTER MAPPING)

dft["text_pattern"] = dft["text_adj"].str.replace("`", "`", regex=False)

dft["text_pattern"] = dft["text_pattern"].str.replace("[A-Z]", "X", regex=True)

dft["text_pattern"] = dft["text_pattern"].str.replace("[a-z]", "x", regex=True)

dft.head().style.set_table_styles(tom_df_style)
# MOST ARE UNIQUE! THIS MIGHT BE PROMISING!

tmp = pd.DataFrame(dft.groupby(["text_pattern"]).size(), columns=["N"])

sns.distplot(tmp["N"], kde=False, rug=True);
# LET'S APPLY THE SAME TRANSFORMATION TO THE LEVEL 1 ENCRYPTED TEXT

dfe1["text_pattern"] = dfe1["ciphertext"].str.replace("`", "`", regex=False)

dfe1["text_pattern"] = dfe1["text_pattern"].str.replace("[A-Z]", "X", regex=True)

dfe1["text_pattern"] = dfe1["text_pattern"].str.replace("[a-z]", "x", regex=True)

dfe1.head().style.set_table_styles(tom_df_style)
# HERE'S ONE EXAMPLE FOUND

tmp1 = dfe1[dfe1["ciphertext_id"]=="ID_d649ebbb2"][["ciphertext","text_pattern"]].rename(columns={"ciphertext":"text"})

tmp2 = dft[dft["plaintext_id"]=="ID_97bea3ff9"][["text_adj","text_pattern"]].rename(columns={"text_adj":"text"})

pd.concat([tmp1,tmp2], ignore_index=True).style.set_table_styles(tom_df_style)
# HERE IS A MESSY (AND VERY SLOW!) ATTEMPT AT "AUTOMATING" THE MATCHING OF SOME ENCRYPTED PHRASES BACK TO THE "BEST GUESS" MATCHING ORIGINAL MESSAGE BASED ON SENTENCE STRUCTURE

results = {}

dft_data = list(dft[["plaintext_id","text_length","text_pattern",]].to_records(index=False))



#eids = ["ID_6100247c5","ID_9bf75d21c","ID_fb906e3a4","ID_93aa4509f","ID_d649ebbb2","ID_4a6fc1ea9","ID_c85d54d74","ID_ac57b8817"]

#for i, dfe1_row in dfe1[dfe1["ciphertext_id"].isin(eids)].iterrows():



# GRAB SOME ARBITRARY RECORDS (LIKE JUST 30 OR SO)

for i, dfe1_row in dfe1.sample(30, random_state=123).iterrows():    

    (ciphertext_id, cipher_text_pattern) = (dfe1_row.ciphertext_id, dfe1_row.text_pattern)

    print(ciphertext_id)

    print("   ETEXT:" + cipher_text_pattern)

    

    results[ciphertext_id] = ""

    maxlength = 0

    for d in dft_data:

        (plaintext_id, plain_text_length, plain_text_pattern) = (d[0], d[1], d[2])

        boxsize = int(math.ceil(plain_text_length / 100.0)) * 100

        startpos = int((boxsize - plain_text_length) / 2)

        if cipher_text_pattern[startpos:startpos+plain_text_length] == plain_text_pattern[startpos:startpos+plain_text_length]:

            if plain_text_length > maxlength:

                results[ciphertext_id] = plaintext_id

                maxlength = plain_text_length

                print("   PTEXT:" + plain_text_pattern)
results
cids = []

tids = []

for k, v in results.items():

    cids.append(k)

    tids.append(v)
# NOW WE CAN THUMB THROUGH SOME MATCHES FOUND AND TRY AND GLEAN SOME MORE INSIGHTS INTO THE MAPPINGS

k = 3



(cid, tid) = (cids[k],tids[k])

tmp1 = dft[dft["plaintext_id"]==tid][["text_adj"]].rename(columns={"text_adj":"text"})

tmp2 = dfe1[dfe1["ciphertext_id"]==cid][["ciphertext"]].rename(columns={"ciphertext":"text"})

pd.concat([tmp1,tmp2], ignore_index=True).style.set_table_styles(tom_df_style)
# FOR MATCH RESULTS FOUND, PAIR PLAIN => ENCYPTED MESSAGES

df_bridge = pd.DataFrame({"TID":tids, "EID":cids})

df_bridge = df_bridge.merge(dft, how="inner", left_on="TID", right_on="plaintext_id")[["EID","plaintext_id","text","text_length"]]

df_bridge = df_bridge.merge(dfe1, how="inner", left_on="EID", right_on="ciphertext_id")[["plaintext_id","text","text_length","ciphertext_id","ciphertext"]]

df_bridge.head(10).style.set_table_styles(tom_df_style)
# LET'S "CLIP" THE CIPHERTEXT TO OVERLAY KEEP THOSE CHARACTERS THAT ACTUALLY ALGN TO THE PLAIN TEXT |....XXXXXXX....| SIMILAR AS HAD DONE ABOVE

list_text     = list(df_bridge["ciphertext"])

list_lenx     = list(df_bridge["text_length"])

list_boxsize  = [int(math.ceil(L / 100.0)) * 100 for L in list_lenx]

list_startpos = list(map(lambda B, L: int((B - L) / 2), list_boxsize, list_lenx))



df_bridge["ciphertext_adj"] = list(map(lambda T, P, L: T[P:P+L], list_text, list_startpos, list_lenx))

df_chars = df_bridge[["text","ciphertext_adj"]]

df_chars.head(10)
# NOW WE CAN ATTEMPT TO MAP ALL char_in => char_out FOR PHRASES INCLUDED IN OUR RESULTS/BRIDGE DATASET

list_chars_in = []

list_chars_out = []

for i in range(0, len(df_chars)):

    chars_in  = list(df_chars.iloc[i,0])

    chars_out = list(df_chars.iloc[i,1])

    assert (len(chars_out) == len(chars_in))

    list_chars_in  = list_chars_in + chars_in

    list_chars_out = list_chars_out + chars_out

    

df_char_map = pd.DataFrame({"in":list_chars_in, "out":list_chars_out})

df_char_map.head(10).style.set_table_styles(tom_df_style)
# LET'S DO A FULL CROSSTAB TO SAY WHAT MAPS TO WHAT AND HOW OFTEN

df_char_cross = pd.pivot_table(pd.DataFrame(df_char_map.groupby(["in","out"]).size(), columns=["N"]), values="N", index=["out"], columns=["in"], aggfunc=np.sum, fill_value=0)

df_char_cross.head(10).style.set_table_styles(tom_df_style)
# LEt'S PLOT THIS

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(df_char_cross, linewidths=1, cmap=sns.light_palette("red"), vmin=0, vmax=5, ax=ax)

ax.xaxis.set_ticks_position("top")
df_chars.head(25).style.set_table_styles(tom_df_style)
# ZOOM IN ON JUST THE lowercare LETTERS AND WE SEE THAT PERHAPS "z" CHARACTERS ARE SIMPLY LEFT ALONE (LIKE PUNCTUATION, CHRACATERS, AND NUMBERS APPEAR TO BE)

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(df_char_cross.iloc[32:,28:], linewidths=1, cmap=sns.light_palette("red"), vmin=0, vmax=5, ax=ax)

ax.xaxis.set_ticks_position("top")
# FIRST ROW CONTAINS ORIGINAL LETTERS WE WANT TO TRANSLATE AND SUBSEQUENT ROWS REPRESENT 1st, 2nd, 3rd, and 4th SHIFT

translation = [

"ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy",

"EFGHIJKLMNOPQRSTUVWXYABCDefghijklmnopqrstuvwxyabcd",

"PQRSTUVWXYABCDEFGHIJKLMNOpqrstuvwxyabcdefghijklmno",

"YABCDEFGHIJKLMNOPQRSTUVWXyabcdefghijklmnopqrstuvwx",

"LMNOPQRSTUVWXYABCDEFGHIJKlmnopqrstuvwxyabcdefghijk",

]

    

# GIVEN AN INPUT CHARACTER AND ITS SEQUENCE MARKER, RETURN ENCYPTED CHARACTER (IF RELEVANT) AND INCREASE SEQUENCE INDICTATOR VALUE (IF RELEVANT)

def decrypt_letter(input_char, s):



    k = s % 4    

    retvals = (input_char, 0)



    if (k == 0):

        k = 4

    

    i = translation[k].find(input_char)

    

    if (i >= 0):

        out_char = translation[0][i]

        retvals = (out_char, 1)



    return retvals
# TEST SOME

print(decrypt_letter("F",1))

print(decrypt_letter("[",53))
# NOW BUILD FUNCTION TO DECRYPT ENTIRE PHRASES

def decrypt_phrase(input_phrase, xstart):

    

    return_chars = []

    x = xstart

    

    for c in input_phrase:

        (out_char, i) = decrypt_letter(c, x)

        x = x + i

        return_chars.append(out_char)

    

    return "".join(return_chars)
# Fks nihslmd hewkd exhnvii lgwj ydfxsdejd:

# But certain issue strokes must arbitrate:

decrypt_phrase("Fks nihslmd hewkd exhnvii lgwj ydfxsdejd:", 1)
# YSHEAPA: Ssi rydhxmlp'i llpxbp edc smi oaxtmnd

# NORFOLK: The cardinal's malice and his potency

decrypt_phrase("YSHEAPA: Ssi rydhxmlp'i llpxbp edc smi oaxtmnd", 4)
# TRIAL AND ERROR SEEMS TO SUGGEST IF WE SEND IN THE FULL CIPHERTEXT, THEN WE CAN TURN THE KNOB TO "2" AND THINGS COME INTO FOCUS (IN THE MIDDLE): 

dfe1.head(15)["ciphertext"].apply(lambda x : decrypt_phrase(x,2))
%%time



# HERE IS A MESSY (AND VERY SLOW!) ATTEMPT AT DECRYPTING LEVEL 1 PHRASES

full_results = []



dfe1_data = list(dfe1[["ciphertext_id","ciphertext"]].to_records(index=False))

dft_id    = list(dft["plaintext_id"])

dft_text  = list(dft["text"])

dft_index = list(dft["index"])



list_lenx     = list(dft["text_length"])

list_boxsize  = [int(math.ceil(L / 100.0)) * 100 for L in list_lenx]

list_startpos = list(map(lambda B, L: int((B - L) / 2), list_boxsize, list_lenx))



# GRAB SOME ARBITRARY RECORDS (LIKE JUST 100 OR SO)

i = 0

N = len(dfe1_data)

for row_enc in dfe1_data:



    i = i + 1

    if i % 1000 == 0:

        print(str(i) + " [" + f"{(i/N):0.2%}" + " ] records processed...")

        

    (ciphertext_id, ciphertext) = (row_enc[0], row_enc[1])

    deciphered_text = decrypt_phrase(ciphertext, 2)

   

    for j in range(0,len(dft_text)):

        (plaintext_id, plain_text, plain_text_index, startpos, plain_text_length) = (dft_id[j], dft_text[j], dft_index[j], list_startpos[j], list_lenx[j])

        if deciphered_text[startpos:startpos+plain_text_length] == plain_text:

            full_results.append([ciphertext_id, ciphertext, plain_text, plaintext_id, plain_text_index])

            break
df_results = pd.DataFrame(full_results, columns=["ciphertext_id","ciphertext","plain_text","plaintext_id","plain_text_index"])

df_results.head().style.set_table_styles(tom_df_style)
# SETUP SUBMISSIONS FILE

df_submissions = dfe[["ciphertext_id"]].merge(df_results[["ciphertext_id","plain_text_index"]], how="left", left_on="ciphertext_id", right_on="ciphertext_id")

df_submissions["plain_text_index"] = df_submissions["plain_text_index"].fillna(0).astype(int)

df_submissions.rename(columns={"plain_text_index": "index"}, inplace=True)

df_submissions.head(10)
# EXPORT SUBMISSION FILE

df_submissions.to_csv("ct3_submission.csv", index=None)