# nothing to install here
import pathlib

import numpy as np

import pandas as pd

import re



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')





# Simplify column names to pandas friendly

_COLUMN_TOKENIZER = re.compile(r'[\w\d]+').findall

def simplify_columns(df, delimiter='_'):

    """ Simplify Column Name in Pandas DataFrame

    

    The method

    

    Arguments

    ---------

    df: pandas.DataFrame or iterable of strings

        The dataframe from which to modify columns. Note that if

        note a dataframe, the method can take lists, tuples, or

        other iterables of strings. In such cases, the output

        will be a list.

    delimiter: str

        As all whitespaces are removed, the delimiter defines the

        string to join tokens from within the original column name.

        This can be set to any string value, but defaults to '_' as

        it is pandas friendly for both `df['col_name']` as well as 

        `df.col_name` approaches.

    """

    

    # get columns from dataframe

    if isinstance(df, pd.DataFrame):

        columns = df.columns

    

    # rename columns

    columns = [

        delimiter.join(_COLUMN_TOKENIZER(col.lower()))

        for col in columns

    ]

    

    # if input was a dataframe modify columns and return full dataframe

    if isinstance(df, pd.DataFrame):

        df.columns = columns

        return df

    return columns





class Colors:

    def __init__(self, red=None, orange=None, yellow=None, green=None, cyan=None, blue=None, violet=None, 

                 white=None, gray=None, black=None):

        # set red value

        if red is None:

            red = '#9e3c3c'

        self.red = red

        

        # set orange value

        if orange is None:

            orange = '#e8882e'

        self.orange = orange

        

        # set yellow value

        if yellow is None:

            yellow = '#e8be43'

        self.yellow = yellow

        

        # set green value

        if green is None:

            green = '#5ea579'

        self.green = green

        

        # set cyan value

        if cyan is None:

            cyan = '#95f4e1'

        self.cyan = cyan

        

        # set blue value

        if blue is None:

            blue = '#4265aa'

        self.blue = blue

        

        # set violet value

        if violet is None:

            violet = '#755082'

        self.violet = violet

        

        # set white value

        if white is None:

            white = '#dae1ea'

        self.white = white

        

        # set gray value

        if gray is None:

            gray = '#616770'

        self.gray = gray

        

        # set black value

        if black is None:

            black = '#25282b'

        self.black = black

        

        

colors = Colors()

labels = ['Red', 'Orange', 'Yellow', 'Green', 'Cyan', 'Blue', 'Violet', 'Gray', 'Black']

for index, color in enumerate(labels):

    plt.bar(

        index * 2, width=2, height=1,

        align='edge',

        color=getattr(colors, color.lower())

    )

plt.xlim(0, 2 * len(labels))

plt.ylim(0, 1)

plt.show()



pd.set_option('display.max_columns', 500)
INPUT_PATH = pathlib.Path('/kaggle/input/CORD-19-research-challenge/2020-03-13/')

WORK_PATH = pathlib.Path('/kaggle/working/')



DATA_PATH = WORK_PATH / 'data'

if not DATA_PATH.is_dir():

    DATA_PATH.mkdir()



!pwd

!ls
!ls /kaggle/input/CORD-19-research-challenge
meta = simplify_columns(pd.read_csv(str(INPUT_PATH / 'all_sources_metadata_2020-03-13.csv')))

display(f"CORD-19 Dataset has {meta.shape[0]:,} records across {meta.shape[1]:,} features.")
meta.columns
meta.head()
full_text_counts = meta.has_full_text.replace({True: 'Full Text', False: 'Limited Text'}).value_counts(normalize=True)



plt.bar(full_text_counts.index[0], full_text_counts.values[0], color=colors.blue, label='Full Text')

plt.bar(full_text_counts.index[1], full_text_counts.values[1], color=colors.red, label='Limited Text')

plt.show()
import uuid

import json





def create_dataset(paper_filename='papers.csv', author_filename='authors.csv', lookup_filename='lookup.csv', overwrite=False):

    

    ### Safety check and prepare output files

    # output file to papers 

    paper_filename = DATA_PATH / paper_filename

    if paper_filename.exists() and not overwrite:

        raise ValueError(f"Papers file `{str(paper_filename)}` exists! Rename or set overwrite to True.")

    paper_filename = str(paper_filename)

        

    # output file to authors 

    author_filename = DATA_PATH / author_filename

    if author_filename.exists() and not overwrite:

        raise ValueError(f"Authors file `{str(author_filename)}` exists! Rename or set overwrite to True.")

    author_filename = str(author_filename)

        

    # output file to lookup

    lookup_filename = DATA_PATH / lookup_filename

    if lookup_filename.exists() and not overwrite:

        raise ValueError(f"Lookup file `{str(lookup_filename)}` exists! Rename or set overwrite to True.")

    lookup_filename = str(lookup_filename)

    

    # Paper Dataframe

    papers = pd.DataFrame(

        columns = [

            'paper_id',

            'title',

            'abstract',

            'text',

            'figure_caption',

        ]

    )

    

    # Author Dataframe

    authors = pd.DataFrame(

        columns = [

            'author_id',

            'full_name',

            'first_name',

            'middle_name',

            'last_name',

            'suffix',

            'laboratory',

            'institution',

            'address_line',

            'post_code',

            'region',

            'country',

            'email',

        ]

    )

    

    # Paper/Author Lookup

    paper_author_lookup = pd.DataFrame(

        columns = [

            'paper_author_id',

            'paper_id',

            'author_id',

        ]

    )

    author_id_lookup = {

        'to_id': {},

        'to_author': {}

    }

    

    for filename in INPUT_PATH.glob('**/*.json'):

        with open(str(filename), 'r') as file:

            

            # Load paper from json file

            paper = json.load(file)

            paper_entry = {}

            

            ### Fill Paper Fields

            # Get paper metadata

            paper_entry['paper_id'] = paper['paper_id']

            paper_entry['title'] = paper['metadata']['title']

            

            # Get paper abstract

            paper_entry['abstract'] = '\n'.join(

                paragraph['text']

                for paragraph in paper['abstract']

            )

            

            # Get paper text

            paper_entry['text'] = '\n'.join(

                paragraph['text']

                for paragraph in paper['body_text']

            )

            

            # Get figure captions

            paper_entry['figure_caption'] = '\n'.join(

                caption['text']

                for caption in paper['back_matter']

            )

            

            ### Fill Author Fields

            for author in paper['metadata']['authors']:

                author_entry = {}

                

                middle_name = ' ' + '-'.join(author['middle']) if len(author['middle']) > 0 else ''

                full_name = f"{author['last']}, {author['first']}{middle_name}{author['suffix']}"

                if full_name not in author_id_lookup['to_id']:

                    author_id = str(uuid.uuid4())

                    author_entry['author_id'] = author_id

                    author_entry['full_name'] = full_name

                    author_entry['first_name'] = author['first']

                    author_entry['middle_name'] = middle_name

                    author_entry['last_name'] = author['last']

                    author_entry['suffix'] = author['suffix']

                    author_entry['email'] = author['email']

                    

                    ### Optional entries

                    # get lab info

                    if 'laboratory' in author['affiliation']:

                        author_entry['laboratory'] = author['affiliation']['laboratory']

                    else:

                        author_entry['laboratory'] = ''

                        

                    # get affiliation info

                    if 'institution' in author['affiliation']:

                        author_entry['institution'] = author['affiliation']['institution']

                    else:

                        author_entry['institution'] = ''

                    

                    # get location information

                    if 'location' in author['affiliation']:

                        # get address 

                        if 'addrLine' in author['affiliation']['location']:

                            author_entry['address_line'] = author['affiliation']['location']['addrLine']

                        else:

                            author_entry['address_line'] = ''



                        # get post/zip code

                        if 'postCode' in author['affiliation']['location']:

                            author_entry['post_code'] = author['affiliation']['location']['postCode']

                        else:

                            author_entry['post_code'] = ''



                        # get region/state

                        if 'region' in author['affiliation']['location']:

                            author_entry['region'] = author['affiliation']['location']['region']

                        else:

                            author_entry['region'] = ''



                        # get country

                        if 'country' in author['affiliation']['location']:

                            author_entry['country'] = author['affiliation']['location']['country']

                        else:

                            author_entry['country'] = ''

                    else:

                        author_entry['address_line'] = ''

                        author_entry['post_code'] = ''

                        author_entry['region'] = ''

                        author_entry['country'] = ''

                    

                    # update author/author_id lookup

                    author_id_lookup['to_id'][full_name] = author_id

                    author_id_lookup['to_author'][author_id] = full_name

                    

                    # append to author dataframe

                    authors = authors.append(author_entry, ignore_index=True)

                else:

                    author_id = author_id_lookup['to_id'][full_name]

                    

                # update paper/author Lookup

                paper_author_lookup_entry = {

                    'paper_author_id': str(uuid.uuid4()),

                    'paper_id': paper['paper_id'],

                    'author_id': author_id,

                }

                

                # append paper/author lookup

                paper_author_lookup = paper_author_lookup.append(paper_author_lookup_entry, ignore_index=True)

                

            # append paper dataframe

            papers = papers.append(paper_entry, ignore_index=True)

        

    # Save files!

    papers.to_csv(paper_filename, index=False)

    authors.to_csv(author_filename, index=False)

    paper_author_lookup.to_csv(lookup_filename, index=False)

        

    display("Completed data cleanup!")
create_dataset(overwrite=True)
pd.read_csv(str(DATA_PATH / 'papers.csv'), nrows=5)