!pip install googletrans



from googletrans import Translator



from dask import bag, diagnostics



import numpy as np 



import pandas as pd 



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import random



import re
train_data = pd.read_csv('/kaggle/input/researchtopictags/train.csv')



test_data = pd.read_csv('/kaggle/input/researchtopictags/test.csv') 
def applyRegexps(text, listRegExp):

    

    """ Applies successively many regexps to a text"""

    

    # apply all the rules in the ruleset

    

    for element in listRegExp:

        

        left = element['left']

        

        right = element['right']

        

        r=re.compile(left)

        

        text=r.sub(right,text)

    

    return text





def detex(latexText):

    

    """Transform a latex text into a simple text"""    

    # initialization

    

    regexps=[]

    

    text=latexText

    

    # remove all the contents of the header, ie everything before the first occurence of "\begin{document}"

    

    text = re.sub(r"(?s).*?(\\begin\{document\})", "", text, 1)

    

    # remove comments

    

    regexps.append({r'left':r'([^\\\d])%.*', 'right':r'\1'})

    

    text= applyRegexps(text, regexps)

    

    regexps=[]

     

    # - replace some LaTeX commands by the contents inside curly rackets

    

    to_reduce = [r'\\emph', r'\\textbf', r'\\textit', r'\\text', r'\\IEEEauthorblockA', r'\\IEEEauthorblockN', r'\\author', r'\\caption',r'\\author',r'\\thanks']

    

    for tag in to_reduce:

        

        regexps.append({'left':tag+r'\{([^\}\{]*)\}', 'right':r'\1'})

    

    text= applyRegexps(text, regexps)

    

    regexps=[]

  

    # - replace some LaTeX commands by the contents inside curly brackets and highlight these contents

    

    to_highlight = [r'\\part[\*]*', r'\\chapter[\*]*', r'\\section[\*]*', r'\\subsection[\*]*', r'\\subsubsection[\*]*', r'\\paragraph[\*]*'];

    

    # highlightment pattern: #--content--#

    

    for tag in to_highlight:

        

        regexps.append({'left':tag+r'\{([^\}\{]*)\}','right':r'\n#--\1--#\n'})

    

    # highlightment pattern: [content]

    

    to_highlight = [r'\\title',r'\\author',r'\\thanks',r'\\cite', r'\\ref'];

    

    for tag in to_highlight:

        

        regexps.append({'left':tag+r'\{([^\}\{]*)\}','right':r'[\1]'})

    

    text= applyRegexps(text, regexps)

    

    regexps=[]

    

    

    # remove LaTeX tags

    # - remove completely some LaTeX commands that take arguments

    to_remove = [r'\\maketitle',r'\\footnote', r'\\centering', r'\\IEEEpeerreviewmaketitle', r'\\includegraphics', r'\\IEEEauthorrefmark', r'\\label', r'\\begin', r'\\end', r'\\big', r'\\right', r'\\left', r'\\documentclass', r'\\usepackage', r'\\bibliographystyle', r'\\bibliography',  r'\\cline', r'\\multicolumn']

    

    # replace tag with options and argument by a single space

    

    for tag in to_remove:

        

        regexps.append({'left':tag+r'(\[[^\]]*\])*(\{[^\}\{]*\})*', 'right':r' '})

    

    text= applyRegexps(text, regexps)

    

    regexps=[]



    

    

    # - replace some LaTeX commands by the contents inside curly rackets

    # replace some symbols by their ascii equivalent

    # - common symbols

    

    regexps.append({'left':r'\\eg(\{\})* *','right':r'e.g., '})

    

    regexps.append({'left':r'\\ldots','right':r'...'})

    

    regexps.append({'left':r'\\Rightarrow','right':r'=>'})

    

    regexps.append({'left':r'\\rightarrow','right':r'->'})

    

    regexps.append({'left':r'\\le','right':r'<='})

    

    regexps.append({'left':r'\\ge','right':r'>'})

    

    regexps.append({'left':r'\\_','right':r'_'})

    

    regexps.append({'left':r'\\\\','right':r'\n'})

    

    regexps.append({'left':r'~','right':r' '})

    

    regexps.append({'left':r'\\&','right':r'&'})

    

    regexps.append({'left':r'\\%','right':r'%'})

    

    regexps.append({'left':r'([^\\])&','right':r'\1\t'})

    

    regexps.append({'left':r'\\item','right':r'\t- '})

    

    regexps.append({'left':r'\\hline[ \t]*\\hline','right':r'============================================='})

    

    regexps.append({'left':r'[ \t]*\\hline','right':r'_____________________________________________'})

    

    # - special letters

    

    regexps.append({'left':r'\\\'{?\{e\}}?','right':r'é'})

    

    regexps.append({'left':r'\\`{?\{a\}}?','right':r'à'})

    

    regexps.append({'left':r'\\\'{?\{o\}}?','right':r'ó'})

    

    regexps.append({'left':r'\\\'{?\{a\}}?','right':r'á'})

    

    # keep untouched the contents of the equations

    

    regexps.append({'left':r'\$(.)\$', 'right':r'\1'})

    

    regexps.append({'left':r'\$([^\$]*)\$', 'right':r'\1'})

    

    # remove the equation symbols ($)

    

    regexps.append({'left':r'([^\\])\$', 'right':r'\1'})

    

    # correct spacing problems

    

    regexps.append({'left':r' +,','right':r','})

    

    regexps.append({'left':r' +','right':r' '})

    

    regexps.append({'left':r' +\)','right':r'\)'})

    

    regexps.append({'left':r'\( +','right':r'\('})

    

    regexps.append({'left':r' +\.','right':r'\.'})    

    

    # remove lonely curly brackets    

    

    regexps.append({'left':r'^([^\{]*)\}', 'right':r'\1'})

    

    regexps.append({'left':r'([^\\])\{([^\}]*)\}','right':r'\1\2'})

    

    regexps.append({'left':r'\\\{','right':r'\{'})

    

    regexps.append({'left':r'\\\}','right':r'\}'})

    

    # strip white space characters at end of line

    

    regexps.append({'left':r'[ \t]*\n','right':r'\n'})

    

    # remove consecutive blank lines

    

    regexps.append({'left':r'([ \t]*\n){3,}','right':r'\n'})

    

    # apply all those regexps

    

    text= applyRegexps(text, regexps)

    

    regexps=[]    

    

    # return the modified text

    

    return text
%%time

train_data['TITLE'] = train_data['TITLE'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))



train_data['ABSTRACT'] = train_data['ABSTRACT'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))
%%time

test_data['TITLE'] = test_data['TITLE'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))



test_data['ABSTRACT'] = test_data['ABSTRACT'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))
train_data.to_csv('train_without_latex_tags.csv',index = False)



test_data.to_csv('test_without_latex_tags.csv',index = False)
train_data = pd.read_csv('/kaggle/input/researchtopictags/train.csv')



test_data = pd.read_csv('/kaggle/input/researchtopictags/test.csv') 
def translate(words, dest):

    

    dest_choices = ['it',

                    'fr',

                    'es',

                    'de',

                    ]

    

    if not dest:

        

        dest = np.random.choice(dest_choices)

        

    translator = Translator()

    

    decoded = translator.translate(words, dest=dest).text

    

    return decoded





def trans_parallel(df, dest):

    

    title_bag = bag.from_sequence(df.TITLE.tolist()).map(translate, dest)

    

    abstract_bag =  bag.from_sequence(df.ABSTRACT.tolist()).map(translate, dest)

    

    with diagnostics.ProgressBar():

        

        titles = title_bag.compute()

        

        abstracts = abstract_bag.compute()

    

    df[['TITLE', 'ABSTRACT']] = list(zip(titles, abstracts))

    

    return df



    

encode_train = train_data.copy().pipe(trans_parallel, dest=None)



decode_train =  encode_train.pipe(trans_parallel, dest='en')



encode_test = test_data.copy().pipe(trans_parallel, dest=None)



decode_test =  encode_test.pipe(trans_parallel, dest='en')

decode_train.to_csv('train_aug_with_latex_tags.csv',index = False)



decode_test.to_csv('test_aug_with_latex_tags.csv',index = False)
%%time

decode_train['TITLE'] = decode_train['TITLE'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))



decode_train['ABSTRACT'] = decode_train['ABSTRACT'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))
%%time

decode_test['TITLE'] = decode_test['TITLE'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))



decode_test['ABSTRACT'] = decode_test['ABSTRACT'].apply(lambda x :detex(x).replace("\n", " ").replace("\\", " "))
decode_train.to_csv('train_aug_without_latex_tags.csv',index = False)



decode_test.to_csv('test_aug_without_latex_tags.csv',index = False)