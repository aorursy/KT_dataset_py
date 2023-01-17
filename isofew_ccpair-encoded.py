!pip install javalang
import pandas as pd
import numpy as np
import javalang
import re
df = pd.read_csv('../input/ccpair/code.csv', lineterminator='\n')
#df[['repo_id', 'file_id']].to_csv('code_id.csv', index=False)
# this is an awful lot of ad-hoc rules, can we do better?
link = re.compile('{@link ([a-zA-Z][a-zA-Z0-9_\.]*)}')
code = re.compile('{@code ([a-zA-Z][a-zA-Z0-9_\.]*)}')
para = re.compile('<p>|</p>')
non_words = re.compile('@[a-z]+|{[^}]*}|<[^>]*>')
dup_space = re.compile('  +')
strips = '/\n\t\r *.'

def replace_punk(s):
    s = ' ' + s + ' '
    # ending puncs
    for p in '\'"$%.,:;?!)':
        s = s.replace(f'{p} ', f' {p} ')
    # starting puncs
    for p in '\'"(':
        s = s.replace(f' {p}', f' {p} ')
    s = s.replace("'s ", " 's ").replace("s' ", "s 's ")
    return s.strip(' ')

def comm_tokenize(s):
    if u'\ufffd' in s.encode().decode('ascii', 'replace'):
        return ''
    s = s.replace('\r', '\n')
    s = link.sub(r'\g<1>', s)
    s = code.sub(r'\g<1>', s)
    s = para.sub('', s)
    t = re.split(r'[ \t]+\*[ \t]+@', s)[0]
    if t.strip(' \n\r\t/*') == '':
        s = re.split(r'[ \t]+\*[ \t]+@[^r][^e]', s)[0]
    else:
        s = t
    first_sent = (s+' ').split('. ')[0]
    first_sent = first_sent.replace('\n', '').replace('*', '').replace('\t', '').strip(strips)
    first_token = first_sent.split(' ')[0]
    if not non_words.match(first_token) or first_token == '@return':
        c = dup_space.sub(' ', non_words.sub('', first_sent)).strip(strips)#.lower()
        d = replace_punk(c).split(' ')
        ret = []
        for e in d:
            if e == '.':
                break
            elif e != '':
                ret.append(e)
        if first_token == '@return' and len(ret) > 0 and ret[0][1:6] != 'eturn':
            ret = ['Returns'] + ret
        return ret
    else:
        return []
def maybe(nothing):
    def decorator(f):
        def g(*a, **b):
            try:
                return f(*a, **b)
            except KeyboardInterrupt as e:
                raise e
            except:
                return nothing
        return g
    return decorator
@maybe([])
def code_tokenize(s):
    return [t.value for t in javalang.parse.tokenize(s)]

@maybe('')
def parse(s):
    return javalang.parse.parse('class X{\n' + s + '\n}').children[2][0].children[4][0]

def traverse(node, pref=''):
    toks = []
    if isinstance(node, javalang.tree.Node):
        toks += [node.__class__.__name__, '(']
        for k in node.attrs:
            v = getattr(node, k)
            if v is not None:
                if isinstance(v, list) or isinstance(v, set):
                    for v_ in v:
                        toks += traverse(v_, pref=k+'/')
                else:
                    toks += traverse(v, pref=k+'/')
        toks += [')', node.__class__.__name__]
    elif str(node) != '':
        toks += [pref + str(node)]
    return toks

def sbt_tokenize(s):
    return traverse(parse(s))
def numeralize(xss):
    vocab = {'〈PAD〉': 0, '〈EOS〉': 1}
    xss_ = []
    for xs in xss:
        xs_ = []
        for x in xs:
            for t in {x, x.split('/')[0]}:
                if t not in vocab:
                    vocab[t] = len(vocab)
            xs_.append(vocab[x])
        xss_.append(xs_)
    return vocab, xss_
def pack(tokenize, xss):
    vocab, xss = numeralize(map(tokenize, xss))
    vector = np.zeros(sum(map(len, xss)), dtype=np.int32)
    starts = np.zeros(len(xss), dtype=np.int32)
    ends = np.zeros(len(xss), dtype=np.int32)
    p = 0
    for i, xs in enumerate(xss):
        vector[p : p + len(xs)] = xs
        starts[i], ends[i] = p, p + len(xs)
        p += len(xs)
    return vocab, vector, starts, ends
%%time
#np.save('comms.npy', pack(comm_tokenize, df.doc))
#np.save('codes.npy', pack(code_tokenize, df.src))
np.save('sbts.npy', pack(sbt_tokenize, df.src))
