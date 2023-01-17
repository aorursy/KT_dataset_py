# --- PACKAGE IMPORT

import os                                      

import re

import sys

import math

import codecs

import itertools



import numpy as np                             # numpy      - working with numbers & math things

import pandas as pd                            # pandas     - excel but way cooler

import seaborn as sns



from tqdm import tqdm                          # tqdm       - for fancy progress bars

from tabulate import tabulate                  # tabulate   - for fancy printing dataframes

import matplotlib.pyplot as plt                # matplotlib - for fancy vis

from IPython.core.display import display, HTML # Ipython    - for fancy notebook sugar



# --- CONFIGURATION



# BASE PATH WHEN USING A KAGGLE KERNEL

kaggle_slug ='../input/kali-2020-usrsharewordlists-directory/KALI2020_usr_share_wordlists'



# BASE PATH WHEN USING KALI 2020

kali_path = '/usr/share/wordlists/'



base = kaggle_slug + kali_path # CHANGE THIS IF RUNNING LOCALLY



# FOR MAKING DEBUGGING FASTER

include_rockyou = True

rockyou_100000s_of_rows = 100



# CASCADING STYLES

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""")
def count_raw_lines(fname):

    # tries to get an idea of how many lines are each file.

    try:

        i = 0

        with open(fname) as f:

            for i, l in enumerate(f):

                pass

        return i + 1

    except:

        try:

            if str(sys.exc_info()[0]) == "<class 'UnicodeDecodeError'>":

                i = 0

                with codecs.open(fname, encoding='latin-1') as f:

                    for i, l in enumerate(f):

                        pass

                return i + 1

            else:

                exit(0)

        except:

            print(sys.exc_info()[0])
def grab_valid_lines(fname):

    # 

    valid_lines = []

    try:

        i = 0

        with open(fname) as f:

            for i, l in enumerate(f):

                if l[0] is not '#': # TODO: check for multiline comments

                    valid_lines.append(l.replace('\n', ''))

                pass

    except:

        try:

            if str(sys.exc_info()[0]) == "<class 'UnicodeDecodeError'>":

                i = 0

                with codecs.open(fname, encoding='latin-1') as f:

                    for i, l in enumerate(f):

                        if l[0] is not '#': # TODO: check for multiline comments

                            valid_lines.append(l.replace('\n', ''))

                        pass

            else:

                exit(0)

        except:

            print(sys.exc_info()[0])

    try:

        valid_lines.remove('')

    except:

        pass

    return valid_lines
keeb_layout = [

    { 'row': 0, 'lo' : '`1234567890-= ',  'up' : '~!@#$%^&*()_+ ' },   # ** NOTES ** 

    { 'row': 1, 'lo' : ' qwertyuiop[]\\', 'up' : ' QWERTYUIOP{}|' },   # Escaped \ and '

    { 'row': 2, 'lo' : ' asdfghjkl;\' ',  'up' : ' ASDFGHJKL:" '  },   # Using a ' ' for padding where the tab, caps, shift, delete & return keys are.

    { 'row': 3, 'lo' : ' zxcvbnm,./ ',    'up' : ' ZXCVBNM<>? '   }    # Typically the left hand handles all 6-keys from the left, the right handles the rest.

]



# TODO: find a way to visualize keyboard keypress sequences with centroids of keys and all that, rather than this ugly matrix-type method
def sequence_analysis(string):

    # Password Keystroke Sequence Analysis 

    

    seq = [ ]

    shift_presses = 0

    #     shift_key = { 'char': 'shift', 'shift' : True, 'pos' : '', 'hand': 'either', 'row' : 3, 'finger': 4 }

    

    for i in range(0, len(string)):

        char = string[i]

#         print(" *** --> " + char + " <-- ***")

        payload = {

            'char'  : char,  # The character

            'shift' : False, # True/False is SHIFT is being held?

            'pos'   : 0,     # How many keys from the left is the key?

            'hand'  : 'R',   # Which hand is doing the typing? 

            'row'   : 0,     # Which row of the qwerty keyboard?

            'finger': 0,     # Which of the 8 fingers is responsible for that key?

            'bear'  : 0      # Bearing of latest action in Radians.

            # TODO: Add 4 Finger Usage (no spacebar, no thumbs, no?)

        }

        for r in range(0, len(keeb_layout)):

            row = keeb_layout[r]

            if char in row['lo']:

                payload['up'] = False

                payload['shift'] = False

                payload['pos'] = row['lo'].find(char)

                payload['row'] = r

                if payload['pos'] <= 5: payload['hand'] = 'L'

#                 print("lowercase row_" + str(r) + " pos_" + str(payload['pos']) + " " + payload['hand'])

            if char in row['up']:

                payload['up'] = True

                payload['pos'] = row['up'].find(char)

                payload['row'] = r

                if payload['pos'] <= 5: payload['hand'] = 'L'

                if len(seq) == 0:

                    payload['shift'] = True

                    shift_presses += 1

                else:

                    if seq[len(seq)-1]['shift'] == True:

                        payload['shift'] = True

                    else:

                        shift_presses += 1

                        payload['shift'] = True

#                 print("uppercase row_", str(r) + " pos_" + str(payload['pos'])  + " " + payload['hand'])



        seq.append(payload)

        

    step_count = len(seq) + shift_presses



    total_manhattan_distance = 0  # TODO: Change these back to lists

    total_xy_distance = 0         # TODO: Change these back to lists

    total_hand_switches = 0   

    current_hand = ''

    bears = [ ]

    

    for b in range(0, len(seq)):

        

        step = seq[b]

        x_1 = step['pos']

        y_1 = step['row']

        

        if b == 0:

            current_hand = step['hand']

        if b > 0:

            x_2 = seq[b-1]['pos']

            y_2 = seq[b-1]['row']



            lat1 = math.radians(x_1)

            lat2 = math.radians(x_2)

            diffLong = math.radians(y_2 - y_1)

            x = math.sin(diffLong) * math.cos(lat2)

            y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

            bears.append(round(math.atan2(x, y)))

            

            total_manhattan_distance += (abs( x_2 - x_1 ) + abs( y_2 - y_1 ))

            total_xy_distance += math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)  

            

            if step['hand'] != current_hand:

                current_hand = step['hand']

                total_hand_switches += 1

    

    try: 

        your_average_bear = round(sum(bears) / len(bears) + 1, 2)

    except:

        your_average_bear = 0

                

    return {

        'switch' : total_hand_switches,         # of times the next key uses your other hand.

        'crow'   : round(total_xy_distance, 2), # Distance Travelled, as the crow flies.

        'man'    : total_manhattan_distance,    # The Manhattan distance.

        'steps'  : step_count,                  # of steps in the sequence, including pressing shift.

        'bear'   : your_average_bear            # Average Compas Bearing In Radians Boo Boo.

    }



# print(sequence_analysis('Spring2017'))
def further_analysis(string):

#     print(string)

    

    return {

        'dated'    : False,  # Is there a date?

        'repeats'  : False,  # Are there repeating characters?

        'eng_word' : False   # Is there an English word?   

    }

# print(further_analysis('Spring2017'))
df_meta = pd.read_csv('../input/kali-2020-usrsharewordlists-directory/wordlists_metadata.csv')



# ********** METADATA NOTES *************

# ========== 'format' COLUMN ============

# * XXX means PASSWORD

# * YYY means USERNAME

# * EXT means EXTENSION

# * SUB means SUBDOMAIN

# * URL means URL

# * PSN means POISONOUS

# * DIR means DIRECTORY

# * PATH means PATH

# * NAME means NAMES

# * WORD means WORD

# * FILE means FILENAME

# * OTHER means TBD/WIP



# ========== 'order' COLUMN ============

# * SENS means ORDERED BY SENSITIVITY

# * FREQ means ORDERED BY FREQUENCY

# * ALPHA means ORDERED APHABETICALLY



display(HTML('<center>' + tabulate(df_meta.head(5), headers='keys', tablefmt='html') + '</center>'))
data = []

cols = [ 

    'file_name', 

    'path_rel', 

    'path_abs',

    'path_kali',

    'file_size', 

    'raw_line_count', 

    'extension', 

    'description',

    'separator',

    'list_type',

    'list_order',

    'row_format'

]



excluded_files = [ 'rockyou.txt' ]

if include_rockyou: excluded_files = [ ]



for dirname, _, filenames in os.walk(base):

    filenames = [f for f in filenames]

    for filename in filenames:

#         print("=== Processing: " + filename)



        if filename not in excluded_files:

        

            ### FILE ENUMERATION 

            path_abs = (dirname + '/' + filename).replace("//","/")

            path_kali = (dirname + '/' + filename).replace(kaggle_slug,"").replace("//","/")

            file_size = os.path.getsize(path_abs)

            extension = filename.split('.')[len(filename.split('.')) - 1]

            raw_line_count = count_raw_lines(path_abs)

            kali_abs_path = path_abs.replace(kaggle_slug, '')



            ### PULL METADATA

            meta_row = df_meta[df_meta['kali_abs_path'] == kali_abs_path ].squeeze()



            ### APPEND SMALL LIST TO BIG LIST

            data.append([

                filename,

                dirname.replace(base,'/'),

                path_abs,

                path_kali,

                file_size,

                raw_line_count,

                extension,

                meta_row['desc'],       # description

                meta_row['separator'],  # separator

                meta_row['type'],       # list_type

                meta_row['order'],      # list_order

                meta_row['format']      # row_format

            ])



df_wordlists = pd.DataFrame(data, columns = cols)



print("=-   Total Files In /wordlists/:  " + str(len(df_wordlists) - 1))
display(HTML(tabulate(df_wordlists.head(5), headers='keys', tablefmt='html')))
df_wordlists['raw_line_count'] = pd.to_numeric(df_wordlists['raw_line_count'].fillna(0), downcast='integer')



first_filter = cols  # TMP, no column filtering yet



df_wordlists_filtered = df_wordlists.filter(items=first_filter)



print("=-   Total Files In /wordlists/:  " + str(len(df_wordlists_filtered)) + "   (from reading all files in the /wordlists/ directory)")



duplicate_filenames = df_wordlists['file_name'].value_counts().loc[ lambda x : x > 1 ].index.tolist()



duplicate_file_indices = []



for duplicate_filename in duplicate_filenames:

    df_duplicates = df_wordlists_filtered[ df_wordlists_filtered['file_name'] == duplicate_filename ]

    unique_filesizes = df_duplicates.drop_duplicates('file_size', keep='first')

    if len(unique_filesizes) == 1:

        duplicate_file_indices.append(unique_filesizes.last_valid_index())



print("=-   Duplicate Files Found:       " + str(len(duplicate_file_indices)) + "    (same filename & number of bytes, different directory)")

   

df_deduped = df_wordlists_filtered.drop(df_wordlists_filtered.index[duplicate_file_indices])



print("=-   Unique Files In /wordlists/: " + str(len(df_deduped)))



df_clean = df_deduped



display(HTML(tabulate(df_clean.head(5), headers='keys', tablefmt='html')))
# ISOLATE ALL FILES CONTAINING PASSWORDS

pass_file_types = [ 

    'pass_common', 

#     'pass_default', 

    'cred_common'

#     'cred_default' 

]



df_pass_files = df_clean[df_clean['list_type'].isin(pass_file_types)]



print("=-  # of Files Containing Passwords:    " + str(len(df_pass_files)))



# FILTER OUT FORMATS I DON'T WANT YET

df_pass_files_formatted = df_pass_files[df_pass_files['row_format'].isin([ 'XXX', 'YYY XXX' ])]



# FILTER OUT FORMATS I DON'T WANT YET

df_pass_files_sorted = df_pass_files_formatted[df_pass_files_formatted['list_order'].isin([ 'freq' ])]



print("=-  # of Pass Files Sorted by Freq:     " + str(len(df_pass_files_sorted)))



display(HTML(tabulate(df_pass_files_sorted.head(5), headers='keys', tablefmt='html')))
# EXTRACT UNIQUE LIST OF ABSOLUTE PATHS

pass_file_list = df_pass_files_sorted['path_abs'].unique()



# BEGIN THE LIST OF BAD PASSWORDS WITH THE WORST PASSWORD IN THE GAME

raw_passlist = [ { 'p': 'password', 'i' : 1 } ]

#            'p' is for password, 'i' is for rank



# LOOP THROUGH EACH FILE AND EXTRACT EACH PASSWORD

for path in pass_file_list:

    file = path.split("/")[len(path.split("/"))-1]

#     print("===- PROCESSING: " + file)

    

    file_passlist = []

    lines = grab_valid_lines(path)

    row_format = str(df_pass_files.iloc[list(pass_file_list).index(path)]['row_format']).replace("\n", "")

    list_order = str(df_pass_files.iloc[list(pass_file_list).index(path)]['list_order']).replace("\n", "")

    target = 'XXX'



    for i in range(0, len(lines)):



        line = lines[i]

        payload = { 'p': 'password', 'i' : 0 }

        if row_format == 'YYY XXX':

            try:

                payload['pass'] = line.split(" ")[1]   # try-catch here because 'no password' is an option sometimes

            except:

                pass

        if row_format == 'XXX': 

            payload['p'] = line

        

        if i + 1 < rockyou_100000s_of_rows * 100000:

            payload['i'] = i + 1



            file_passlist.append(payload)



    raw_passlist.extend(file_passlist)



print("=-  # of Raw Passwords Extracted:       " + f"{len(raw_passlist):,}" )



seen_passes = set()

deduped_passlist = []

for obj in raw_passlist:

    if obj['p'] not in seen_passes:

        deduped_passlist.append(obj)

        seen_passes.add(obj['p'])



print("=-  # of Duplicate Passwords Removed:   -" + f"{len(raw_passlist) - len(deduped_passlist):,}" )



print("=-  # of Unique Passwords Extracted:    " + f"{len(deduped_passlist):,}" )
#### Look at me.



del seen_passes

del raw_passlist

del file_passlist

del duplicate_filenames

del duplicate_file_indices



del df_deduped

del df_wordlists

del df_duplicates

del df_pass_files

del df_pass_files_sorted

del df_wordlists_filtered

del df_pass_files_formatted



analysis_passlist = deduped_passlist

del deduped_passlist



#### I am the garbage collector now.
engineered_passlist = [  ]



for i in tqdm(range(0, len(analysis_passlist))):

    pw_obj = analysis_passlist[i]

    pw = pw_obj['p']



    # BASIC STRING ANALYSIS

    pw_char_count = len(pw)

    pw_num_count = len(re.sub("[^0-9]", "", pw))

    pw_cap_count = len(re.sub("[^A-Z]", "", pw))

    pw_low_count = len(re.sub("[^a-z]", "", pw))

    pw_sym_count = pw_char_count - ( pw_num_count + pw_cap_count + pw_low_count )

    

    # KEYSTROKE SEQUENCE ANALYSIS

    key_seq_ana = sequence_analysis(pw)

    

    # RANKNING (pertinant to frequency-ordered password lists, which save time)

#     pw_rank = pw_obj['i']



    # JOIN THE OTHERS

    engineered_passlist.append({

        'pw'         : pw,

        'char_cnt'   : pw_char_count,

        'num_cnt'    : pw_num_count,

        'cap_cnt'    : pw_cap_count,

        'low_cnt'    : pw_low_count,

        'sym_cnt'    : pw_sym_count,

        'switches'   : key_seq_ana['switch'],

        'xy_dist'    : key_seq_ana['crow'],

        'man_dist'   : key_seq_ana['man'],

        'steps'      : key_seq_ana['steps'],

        'bear'       : key_seq_ana['bear'],

        'rank'       : pw_obj['i']

    })

    

del analysis_passlist # 'You have run out of memory' // Is this big data?
df_passwords = pd.DataFrame(engineered_passlist)



del engineered_passlist # The ol' switcheroo
df_passwords = df_passwords[df_passwords.index < int( len(df_passwords) / 2 )]  



print("=- Working with: " + str(len(df_passwords)) + " Unique Passwords")



# display(HTML(tabulate(df_passwords.sort_values(by=['rank'],ascending=False).head(5), headers='keys', tablefmt='html')))

display(HTML('<center>' + tabulate(df_passwords.head(5), headers='keys', tablefmt='html')+'<center>'))
plt.figure(figsize=(16,13))



plt.rcParams.update({ "figure.facecolor": (0, 0, 0, 0), "axes.facecolor": (0, 0, 0, 0) })



sns.heatmap(

    df_passwords.corr(), 

    annot = True, 

    center = 0,  

    cmap = "Spectral", 

    mask = np.triu(np.ones_like(df_passwords.corr(), dtype=bool))

).set_title('A corrplot using the features engineered above.')



plt.show()
for variable in df_passwords.columns:

    

    if variable not in [ 'pw', 'rank' ] :

        

        plt.figure(figsize=(16,4))

        

        plt.hist(

            df_passwords[variable], 

            color = 'blue', 

            edgecolor = 'black', 

            bins = 500

        )



        sns.distplot(

            df_passwords[variable], 

            hist=True, 

            kde=False, 

            bins=69, 

            color = 'blue',

            hist_kws={'edgecolor':'black'}

        )



        plt.title('Frequency Inspection For: ' + variable)

        plt.ylabel('Frequency')

        plt.xlabel(variable)



        plt.show()
# for index, row in df_meta.iterrows():

#     display(HTML('<div style="border:2px solid black;height:275px;border-radius:4px;padding:6px;margin:5px;">' +

#                       '<h1>' + row['file_name'] + '</h1>' +

#                       '<p>TYPE: ' + row['type'] + '</p>' +

#                       '<a href="' + row['file_name'] + '">INFO</a>' + 

#                       '<p>SOURCE: ' + row['src'] + '</p>' + 

#                       '<p>' + row['desc'] + '</p>' + 

#                  '</div>'

#                 ))
# this is where I'll take all the stuff and put it into organized buckets.



# * Maybe something like this? : 

# ```

# /ettu/cred/                   # CREDENTIAL COMBOS (USER&nbsp;PASS\n)

#         /default/ 

#             all.txt 

#             /oracle/

#             /apache/

#             /google/

#             ...

# /ettu/pass/                   # PASSWORDS (pass\n)

#         /admin/ --> sml.txt, med.txt, lrg.txt

#         /users/ --> sml.txt, med.txt, lrg.txt

# /ettu/dir/  # DIRECTORY NAMES

#         /www/

#         /cms/

# /ettu/psn/  # POISON (injection/attack strings)

#         /inj/

#             /sql/

#             /web/

# ```



# * Output an intuitive file-structure for a reorganized /wordlists/ directory for the next time I do the OSCP

# * Maybe saved to /opt/ or /etc/ or just /tmp/...?

# * Add all new TLDs to a list

# * Add other alphabets
# ¯\_(ツ)_/¯