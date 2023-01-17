%load_ext autoreload
%autoreload 2
%matplotlib inline
from pathlib import Path
import zipfile
import pandas as pd
import seaborn as sns
import requests
from PIL import Image
from io import BytesIO
from functools import partial
sns.set_style('whitegrid')
sns.set_context('notebook')
# path = Path('recognition')
path = Path('..')/'input'
# !kaggle competitions download -c landmark-recognition-challenge -p {path}
# for f in path.iterdir():
#     zipfile.ZipFile(f).extractall(path=path)
!ls {path}
!head -5 {path}/train.csv
!head -5 {path}/test.csv
!head -5 {path}/sample_submission.csv
df = pd.read_csv(path/'train.csv')
df.head()
df.info()
df.apply(lambda col: col.duplicated().any())
df.landmark_id.nunique(), df.landmark_id.min(), df.landmark_id.max()
df.landmark_id.value_counts().reset_index(drop=True).plot(logy=True);
df['url_domain'] = df.url.str.split('/').str[2]
df.url_domain.nunique()
df.url_domain.value_counts(ascending=True).plot.barh(title='Full domains');
df['url_sld'] = df.url_domain.str.split('.').str[-2:].str.join('.')
df.url_sld.value_counts(ascending=True).plot.barh(title='Second-level domains');
# The first 3 slashes belong to protocol spec and domain name.
df['url_path'] = df.url.str.split('/').str[3:].str.join('/')
# Assume that the URL "http://example.com/p1" has 1 path part.
df['url_pathparts'] = df.url_path.str.count('/') + 1
df.url_pathparts.value_counts(sort=False)
def listurls(nparts, network=True, n=None):
    """Print landmark ID, number of items with the same landmark ID, response code and URL."""
    network = False  # NOTE: Remove this line when running the code on a machine with network access.
    print('{:>6} {:>6} {:>6} {}'.format('LMID', 'COUNT', 'RESP', 'URL'))
    df_filtered = df.loc[df.url_pathparts == nparts, ['url', 'landmark_id']]
    if n is not None:
        df_filtered = df_filtered[:n]
    for row in df_filtered.itertuples():
        lid = row.landmark_id
        count = len(df[df.landmark_id == lid])
        if network:
            try:
                code = requests.head(row.url, allow_redirects=True, timeout=1).status_code
            except requests.exceptions.RequestException:
                code = 'FAIL'
        else:
            code = 'SKIP'
        print(f'{lid:>6} {count:>6} {code:>6} {row.url}')
def urlmatches(nparts, regex):
    """True if all URLs match the regex."""
    return df.url_path[df.url_pathparts == nparts].str.match(regex).all()
def listnonmatches(nparts, regex, n=5):
    """DataFrame with URLs that don't match the regex (limit size with n)"""
    return df.url_path[df.url_pathparts == nparts].pipe(lambda up: up[~up.str.match(regex)][:n])
def urlmatchcounts(nparts, regex, group=0):
    """Match the regex and print counts of identical match groups."""
    return df.url_path[df.url_pathparts == nparts].str.extract(regex, expand=True)[group].value_counts()
def showimage(url):
    """Print the image's (width, height) and display it."""
    print('NO NETWORK'); return None  # NOTE: Remove this line to actually see the images.
    img = Image.open(BytesIO(requests.get(url).content))
    print(img.size)
    return img
listurls(1, network=False, n=5)
urlmatches(1, r'[\w\d_-]+(?:%3Dw\d+-h\d+-no)?')
df.url[df.url_pathparts == 1].iloc[0]
showimage('https://lh3.googleusercontent.com/xV1jw21l7RxwdEkLhKNxBDn0hox29kT2XYPLb3vnfw')
showimage('https://lh3.googleusercontent.com/xV1jw21l7RxwdEkLhKNxBDn0hox29kT2XYPLb3vnfw' + '%3Dw100')
showimage('https://lh3.googleusercontent.com/xV1jw21l7RxwdEkLhKNxBDn0hox29kT2XYPLb3vnfw' + '%3Dh100')
showimage('https://lh3.googleusercontent.com/xV1jw21l7RxwdEkLhKNxBDn0hox29kT2XYPLb3vnfw' + '%3Dw100-h50')
showimage('https://lh3.googleusercontent.com/xV1jw21l7RxwdEkLhKNxBDn0hox29kT2XYPLb3vnfw' + '%3Dw1000-h1000')
listurls(2)
listurls(5)
listurls(7)
listurls(9)
listurls(11)
listurls(3, network=False, n=5)
urlmatches(3, r'photos/[\w-]+/\d+\.jpg')
urlmatchcounts(3, r'photos/([\w-]+)/\d+\.jpg')
listurls(4, network=False, n=5)
urlmatches(4, r'(?:mw-panoramio|static\.panoramio\.com)/photos/[\w-]+/\d+\.jpg')
urlmatchcounts(4, r'(?:mw-panoramio|static\.panoramio\.com)/photos/([\w-]+)/\d+\.jpg')
def listsizecategories(urlfmt, cats):
    print('NO NETWORK'); return  # NOTE: Remove this line to actually see the size categories.
    for cat in cats:
        try:
            url = urlfmt.format(cat)
            sz = Image.open(BytesIO(requests.get(url).content)).size
        except requests.exceptions.RequestException:
            sz = 'FAIL'
        print('{:>20}: {}'.format(cat, sz))
listsizecategories('http://static.panoramio.com/photos/{}/70761397.jpg',
                   '''original large 1920x1280 medium small thumbnail
                      iw-thumbnail square mini_square'''.split())
listurls(6, network=False, n=5)
urlmatches(6, r'(?:[\w\d.-]+/){4}(?:[\w\d%-])*/')
urlmatchcounts(6, r'(?:[\w\d.-]+/){4}([\w\d%-]*)/')[:10]
urlmatchcounts(6, r'(?:[\w\d.-]+/){4}([a-z]+)(?:[\w\d%-])*/')
listsizecategories('http://lh6.ggpht.com/-vKr5G5MEusk/SR6r6SJi6mI/AAAAAAAAAco/-7JrhF1dfso/{}/',
                   's100 w100 h100 rj d'.split())
listsizecategories('https://lh3.googleusercontent.com/-LOW2cjAqubA/RvE11dfgUaI/AAAAAAAABoU/ItwXEejtwHg/{}/',
                   's200 w200 h200 rj d'.split())
def resizeurl(url, minsize, sizecategory):
    parts = url.split('/')
    # As before, don't count protocol spec and domain name slashes.
    nparts = len(parts) - 3
    if nparts == 1:
        before = parts[-1].partition('%')[0]
        after = f'3Dw{minsize}-h{minsize}'
        parts[-1] = before + '%' + after
    elif nparts == 3 or nparts == 4:
        parts[-2] = sizecategory
    elif nparts == 6:
        parts[-2] = f's{minsize}'
    else:
        return None
    return '/'.join(parts)
df['url_resized'] = df.url.transform(partial(resizeurl, minsize=500, sizecategory='large'))
df.url_resized.isnull().sum()
(df
 .dropna()
 .drop(columns=['url'])
 .rename(columns=dict(url_resized='url'))
 .to_csv('trainsmall.csv', index=False, columns=['id', 'url', 'landmark_id']))
!head -5 trainsmall.csv