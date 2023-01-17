# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import csv

#

#

#with open('some.csv', newline='', encoding='utf-8') as f:

with open( "../input/" + 'atussum.csv', newline='' ) as csv_File:

    #

    dialect = csv.Sniffer().sniff( csv_File.read(1024) )

    print( "sniff.dialect:", dialect )   

    # back to start

    csv_File.seek(0)

    #reader = csv.reader( csvfile, dialect )

    #

    # csv.reader(csvfile, dialect='excel', **fmtparams)

    csv_Reader = csv.reader( 

        csv_File, 

        dialect

        #delimiter=' ', 

        #quotechar='|' 

        #quoting=csv.QUOTE_NONE

    )

    print( "csv_Reader.dialect:", csv_Reader.dialect )   

    #next( csv_Reader )

    for row in csv_Reader:

        if csv_Reader.line_num < 3:

            print(', '.join( row ))

        else:

            break # for

    #    

    # csv.DictReader(f, fieldnames=None, restkey=None, restval=None, dialect='excel', 

    #   *args, **kwds)

    csv_File.seek(0)

    # from current 'csv_File' position

    dict_Reader = csv.DictReader( csv_File ) 

    #next( dict_Reader )

    #for row in reader:

    #    print( row['first_name'], row['last_name'] )

    for i in range( 0, 3, 1 ):

        row = next( dict_Reader ) 

        print( "row:", row )
import pandas.io.date_converters as conv

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#

#

# pandas.read_csv(filepath_or_buffer, sep=', ', delimiter=None, 

#   header='infer', names=None, index_col=None, usecols=None, ...

# Read CSV (comma-separated) file into DataFrame

# Also supports optionally iterating 

# or breaking of the file into chunks.

# pd.read_csv(StringIO(data), usecols=lambda x: x.upper() in ['COL1', 'COL3'])

# pd.read_csv(StringIO(data), skiprows=lambda x: x % 2 != 0)

#

# Iteration

#    iterator : boolean, default False

#        Return TextFileReader object for iteration 

#        or getting chunks with get_chunk().

# Iterating through files chunk by chunk

# table = pd.read_table('tmp.sv', sep='|')

# reader = pd.read_table('tmp.sv', sep='|', chunksize=4)

# <pandas.io.parsers.TextFileReader at 0x7fbbcdac6588>

# for chunk in reader:

#     print(chunk)

# reader = pd.read_table('tmp.sv', sep='|', iterator=True)

# reader.get_chunk(5)

#

# df = pd.read_csv('https://download.bls.gov/pub/time.series/cu/cu.item', sep='\t')

# df = pd.read_csv('s3://pandas-test/tips.csv')

#

# you can indicate the data type for the whole DataFrame 

# or individual columns:

# df = pd.read_csv(StringIO(data), dtype={'b': object, 'c': np.float64})

# df = pd.read_csv('tmp.csv', header=None, parse_dates=date_spec,

#   date_parser=conv.parse_date_time)

# pd.read_csv(StringIO(data), comment='#', skiprows=2)

# dia = csv.excel()

# dia.quoting = csv.QUOTE_NONE

# pd.read_csv(StringIO(data), dialect=dia)

# Automatically “sniffing” the delimiter

# pd.read_csv('tmp2.sv', sep=None, engine='python')

pandas_Reader = pd.read_csv( 

    filepath_or_buffer = "../input/atussum.csv", 

    # usecols : array-like or callable

    usecols = [

        'tucaseid', 'gemetsta', 'gtmetsta', 'peeduca', 'pehspnon', 'ptdtrace', 

        'teage', 'telfs', 'temjot', 'teschenr', 'teschlvl', 'tesex', 'tespempnot', 

        'trchildnum', 'trdpftpt', 'trernwa', 'trholiday', 'trspftpt', 'trsppres', 

        'tryhhchild', 'tudiaryday', 'tufnwgtp', 'tehruslt', 'tuyear'

    ],

    iterator = True,

    chunksize = 1

)

print( ".get_chunk(0):", pandas_Reader.get_chunk(0) )

pandas_Reader.get_chunk(1)
import pandas.io.date_converters as conv

#import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#

#

# DataFrame.head([n])	Returns first n rows

df = pd.read_csv(

    filepath_or_buffer = "../input/atussum.csv", 

    usecols = [

        'tucaseid', 'gemetsta', 'gtmetsta', 'peeduca', 'pehspnon', 'ptdtrace', 

        'teage', 'telfs', 'temjot', 'teschenr', 'teschlvl', 'tesex', 'tespempnot', 

        'trchildnum', 'trdpftpt', 'trernwa', 'trholiday', 'trspftpt', 'trsppres', 

        'tryhhchild', 'tudiaryday', 'tufnwgtp', 'tehruslt', 'tuyear'

    ]

)

df.head( 3 )