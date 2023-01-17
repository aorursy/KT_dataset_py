import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline

import warnings

warnings.filterwarnings('ignore')



from sklearn.decomposition import PCA, kernel_pca

from sklearn.manifold import TSNE

from tqdm import tqdm



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../input/program_codes/code/"]).decode("utf8"))
base = '../input/program_codes/code/'

# READ Stuff

sols = pd.read_csv('../input/solutions.csv').dropna()

first, second, third = [pd.read_csv(base+i+'.csv') for i in 'first,second,third'.split(',')]

# Merge stuff

code = pd.concat([first, second, third])

df = sols.merge(code, how='left', on='SolutionID')

del(sols);del(code);del(first);del(second);del(third)

# Info

df.info()
# We only use what we need. So we drop a few columns

df = df[['QCode', 'SolutionID','Solutions', 'Status', 'Language']]

df.Language = df.Language.str.split(' ').str[0]

df = df.loc[df.Language == 'C']

df.info()
# http://tigcc.ticalc.org/doc/keywords.html

# https://www.programiz.com/python-programming/keyword-list

# http://cs.smu.ca/~porter/csc/ref/cpp_keywords.html

# 

C_keys = '''auto,break,case,char,const,continue,default,do,double,else,enum,extern,float,

for,goto,if,int,long,register,return,short,signed,sizeof,static,struct,switch,typedef,

union,unsigned,void,volatile,while'''.replace('\n', '').split(',')

 # FOR FUTURE USE we keep these keys too

Cpp_keys = '''auto,const,double,float,int,short,struct,unsigned,break,continue,

else,for,long,signed,switch,void,case,default,enum,goto,register,sizeof,typedef,

volatile,char,do,extern,if,return,static,union,while,asm,dynamic_cast,namespace,

reinterpret_cast,try,bool,explicit,new,static_cast,typeid,catch,false,operator,

template,typename,class,friend,private,this,using,const_cast,inline,public,throw,

virtual,delete,mutable,protected,true,wchar_t,and,bitand,compl,not_eq,or_eq,xor_eq,

and_eq,bitor,not,or,xor'''.replace('\n', '').split(',')



java_keys = '''

abstract,continue,for,new,switch,assert,default,goto,package,synchronized,boolean,

do,if,private,this,break,double,implements,protected,throw,byte,else,import,public,

throws,case,enum,instanceof,return,transient,catch,extends,int,short,try,char,final,

interface,static,void,class,finally,long,strictfp,volatile,const,float,native,super,while

'''.replace('\n', '').split(',')



python_keys = '''False,class,finally,is,return,None,continue,for,lambda,try,True,def,

from,nonlocal,while,and,del,global,not,with,as,elif,if,or,yield,assert,else,

import,pass,break,except,in,raise'''.replace('\n', '').split(',')
def make_structure_cleaner(keys):

    def cleaner_function(code):

        if isinstance(code, str):

            special = '`1234567890-=~!@#$%^&*()_+[]{}\|;:",./<>?' + "'"

            for sp in special:

                code = code.replace(sp, ' ' + sp + ' ')

            words = code.replace('\n', ' ').split(' ')

            needed_words = keys + list(special)

            structure = ' '.join([w for w in words if w in needed_words])

            return structure

        else:

            return None

    return cleaner_function

print('Cleaner made')

c_cleaner = make_structure_cleaner(C_keys)
print('Applying to Solutions')

c_structs = c_code.Solutions.apply(c_cleaner)

print('Done')
