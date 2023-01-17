# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import praw



r = praw.Reddit(

    user_agent='',

    client_id ='',

    client_secret='') 
subreddits = [    #Note I am excluding r/announcements 

    'funny',

    'AskReddit',

    'gaming', 

    'pics',

    'science',

    'worldnews', 

    'aww', 

    'todayilearned', 

    'movies'

]
import random



posts = []

for theSub in subreddits:

    sub = r.subreddit(theSub)

    top_posts = [post for post in sub.top('all')]

    posts.append(top_posts)

posts = np.array(posts)
worthy_comments = []
worthy_comments = [praw.models.Comment(r,id='ebkvyoq'), praw.models.Comment(r,id='ebkvi44'), praw.models.Comment(r,id='ebl427t'), praw.models.Comment(r,id='ebljei8'), praw.models.Comment(r,id='eblnhmd'), praw.models.Comment(r,id='efped3o'), praw.models.Comment(r,id='efponqa'), praw.models.Comment(r,id='efpo7m0'), praw.models.Comment(r,id='efpg2la'), praw.models.Comment(r,id='den9n1k'), praw.models.Comment(r,id='djfea07'), praw.models.Comment(r,id='djfeg2j'), praw.models.Comment(r,id='djffg7j'), praw.models.Comment(r,id='ei7oaxh'), praw.models.Comment(r,id='ei7nanv'), praw.models.Comment(r,id='ei7n7z5'), praw.models.Comment(r,id='ei7t8jd'), praw.models.Comment(r,id='djms6y2'), praw.models.Comment(r,id='dsqviip'), praw.models.Comment(r,id='dr9zt7c'), praw.models.Comment(r,id='e56hmmf'), praw.models.Comment(r,id='dg2wzzr'), praw.models.Comment(r,id='dg2ts4t'), praw.models.Comment(r,id='evfqyja'), praw.models.Comment(r,id='evfqs1i'), praw.models.Comment(r,id='datjhua'), praw.models.Comment(r,id='datioho'), praw.models.Comment(r,id='esetxs5'), praw.models.Comment(r,id='esg3whj'), praw.models.Comment(r,id='eiszp8l'), praw.models.Comment(r,id='eit4n6y'), praw.models.Comment(r,id='eit04aa'), praw.models.Comment(r,id='eit124n'), praw.models.Comment(r,id='eit0blp'), praw.models.Comment(r,id='eit0yga'), praw.models.Comment(r,id='eiszkrs'), praw.models.Comment(r,id='eisxz0z'), praw.models.Comment(r,id='eit3aa4'), praw.models.Comment(r,id='eisxv1k'), praw.models.Comment(r,id='evxg2ci'), praw.models.Comment(r,id='evx9eux'), praw.models.Comment(r,id='evxdbau'), praw.models.Comment(r,id='evxhzxg'), praw.models.Comment(r,id='evxa9la'), praw.models.Comment(r,id='evxhsd9'), praw.models.Comment(r,id='evx8ujn'), praw.models.Comment(r,id='evxb95i'), praw.models.Comment(r,id='evx6otu'), praw.models.Comment(r,id='evxg65e'), praw.models.Comment(r,id='evxi8ei'), praw.models.Comment(r,id='evxbbkh'), praw.models.Comment(r,id='evx56pk'), praw.models.Comment(r,id='evxejp8'), praw.models.Comment(r,id='evxbyey'), praw.models.Comment(r,id='evx9a2t'), praw.models.Comment(r,id='evx9bpm'), praw.models.Comment(r,id='evxcvky'), praw.models.Comment(r,id='evxgl0i'), praw.models.Comment(r,id='evxeiq0'), praw.models.Comment(r,id='e5vf8zf'), praw.models.Comment(r,id='e5vcv5b'), praw.models.Comment(r,id='dsw4aay'), praw.models.Comment(r,id='dsvm02s'), praw.models.Comment(r,id='dsvlu8r'), praw.models.Comment(r,id='dsvmmhr'), praw.models.Comment(r,id='ece8wdm'), praw.models.Comment(r,id='ece30zf'), praw.models.Comment(r,id='ece9ikk'), praw.models.Comment(r,id='ece63xp'), praw.models.Comment(r,id='ecdpi8f'), praw.models.Comment(r,id='ecduz8z'), praw.models.Comment(r,id='ecdvd05'), praw.models.Comment(r,id='ece3qlo'), praw.models.Comment(r,id='ece4zdj'), praw.models.Comment(r,id='ecdzsff'), praw.models.Comment(r,id='ecdoo62'), praw.models.Comment(r,id='e46g07q'), praw.models.Comment(r,id='e46hcht'), praw.models.Comment(r,id='e46p7ii'), praw.models.Comment(r,id='e46e1d5'), praw.models.Comment(r,id='e46j9bi'), praw.models.Comment(r,id='ebliyz9'), praw.models.Comment(r,id='eblw053'), praw.models.Comment(r,id='exht0dl'), praw.models.Comment(r,id='exhodrr'), praw.models.Comment(r,id='exi0ven'), praw.models.Comment(r,id='exhtpr6'), praw.models.Comment(r,id='exhshaq'), praw.models.Comment(r,id='exhw3s8'), praw.models.Comment(r,id='exic2ss'), praw.models.Comment(r,id='eh2k7ov'), praw.models.Comment(r,id='eh2sqnc'), praw.models.Comment(r,id='eh2tsmr'), praw.models.Comment(r,id='eh2h3eo'), praw.models.Comment(r,id='eh2vkri'), praw.models.Comment(r,id='eh2ffj8'), praw.models.Comment(r,id='eh2c06t'), praw.models.Comment(r,id='eh2bzvr'), praw.models.Comment(r,id='eh2cqkg'), praw.models.Comment(r,id='eh2bbwu'), praw.models.Comment(r,id='eh2drug'), praw.models.Comment(r,id='eh2eh3d'), praw.models.Comment(r,id='eh2ax1r'), praw.models.Comment(r,id='eh2g59s'), praw.models.Comment(r,id='eh2b6zh'), praw.models.Comment(r,id='eh2f4cw'), praw.models.Comment(r,id='eh2ee98'), praw.models.Comment(r,id='eh27dad'), praw.models.Comment(r,id='eh27m0i'), praw.models.Comment(r,id='eh2eape'), praw.models.Comment(r,id='eh2afdp'), praw.models.Comment(r,id='eh2fxpr'), praw.models.Comment(r,id='eh26860'), praw.models.Comment(r,id='eh2b867'), praw.models.Comment(r,id='eh2ckgc'), praw.models.Comment(r,id='eh2bi6s'), praw.models.Comment(r,id='eh2d6it'), praw.models.Comment(r,id='eh26rmw'), praw.models.Comment(r,id='eh2a822'), praw.models.Comment(r,id='eh2gom4'), praw.models.Comment(r,id='eh263qx'), praw.models.Comment(r,id='eh257e7'), praw.models.Comment(r,id='eh2iirt'), praw.models.Comment(r,id='eh25q9q'), praw.models.Comment(r,id='eh291f3'), praw.models.Comment(r,id='eh2alhy'), praw.models.Comment(r,id='eh2ggmn'), praw.models.Comment(r,id='ecixxq7'), praw.models.Comment(r,id='ecip4vq'), praw.models.Comment(r,id='ecit3g1'), praw.models.Comment(r,id='ecio7p4'), praw.models.Comment(r,id='eciqag2'), praw.models.Comment(r,id='ecit2yz'), praw.models.Comment(r,id='eciqjxh'), praw.models.Comment(r,id='ecio6k6'), praw.models.Comment(r,id='eciktlu'), praw.models.Comment(r,id='ecipdvz'), praw.models.Comment(r,id='ecifzrk'), praw.models.Comment(r,id='ecirfri'), praw.models.Comment(r,id='eciu6bl'), praw.models.Comment(r,id='eciqkj2'), praw.models.Comment(r,id='eciogai'), praw.models.Comment(r,id='eciqu9d'), praw.models.Comment(r,id='ecit7t9'), praw.models.Comment(r,id='eciiox9'), praw.models.Comment(r,id='ecio7x5'), praw.models.Comment(r,id='eciur1y'), praw.models.Comment(r,id='ecirmia'), praw.models.Comment(r,id='eciwhpt'), praw.models.Comment(r,id='ecivmmp'), praw.models.Comment(r,id='ecih4da'), praw.models.Comment(r,id='eciveuk'), praw.models.Comment(r,id='ecixoxb'), praw.models.Comment(r,id='ecix9ml'), praw.models.Comment(r,id='eciyr50'), praw.models.Comment(r,id='eciyxem'), praw.models.Comment(r,id='en27b8v'), praw.models.Comment(r,id='en22x9q'), praw.models.Comment(r,id='en21ky7'), praw.models.Comment(r,id='en22jzh'), praw.models.Comment(r,id='en2a4k8'), praw.models.Comment(r,id='eguah99'), praw.models.Comment(r,id='egu4gef'), praw.models.Comment(r,id='egu6v6w'), praw.models.Comment(r,id='egtz9po'), praw.models.Comment(r,id='egu0fp0'), praw.models.Comment(r,id='egu4egw'), praw.models.Comment(r,id='egu49bj'), praw.models.Comment(r,id='egtsikn'), praw.models.Comment(r,id='egu2bc8'), praw.models.Comment(r,id='egu1yhh'), praw.models.Comment(r,id='egtukxw'), praw.models.Comment(r,id='egu2hhy'), praw.models.Comment(r,id='egu1joi'), praw.models.Comment(r,id='egu7oxh'), praw.models.Comment(r,id='egtw8ff'), praw.models.Comment(r,id='egtxtsd'), praw.models.Comment(r,id='egtxtec'), praw.models.Comment(r,id='egu0ov9'), praw.models.Comment(r,id='ekft6cz'), praw.models.Comment(r,id='ekg22x9'), praw.models.Comment(r,id='ekfl9lg'), praw.models.Comment(r,id='ekfpw2z'), praw.models.Comment(r,id='ekfjetq'), praw.models.Comment(r,id='ekfkxig'), praw.models.Comment(r,id='ekfhlm4'), praw.models.Comment(r,id='dxfe2ql'), praw.models.Comment(r,id='f2d9304'), praw.models.Comment(r,id='f2cscbk'), praw.models.Comment(r,id='f2cn2cq'), praw.models.Comment(r,id='f2cusyt'), praw.models.Comment(r,id='f2d2xqi'), praw.models.Comment(r,id='f2cubyv'), praw.models.Comment(r,id='f2d3w6c'), praw.models.Comment(r,id='f2cpc4j'), praw.models.Comment(r,id='ekokfjf'), praw.models.Comment(r,id='ekoinr6'), praw.models.Comment(r,id='ekodzxx'), praw.models.Comment(r,id='ekof9su'), praw.models.Comment(r,id='ekocp9s'), praw.models.Comment(r,id='ekodnqm'), praw.models.Comment(r,id='ecr4l67'), praw.models.Comment(r,id='ecquc43'), praw.models.Comment(r,id='ecqs3hd'), praw.models.Comment(r,id='ecqwjh4'), praw.models.Comment(r,id='ecqnp6n'), praw.models.Comment(r,id='ecr4t03'), praw.models.Comment(r,id='ecqod1i'), praw.models.Comment(r,id='ecqskmu'), praw.models.Comment(r,id='ecr8xbz'), praw.models.Comment(r,id='ecqtuld'), praw.models.Comment(r,id='ewvskob'), praw.models.Comment(r,id='ewvrytm'), praw.models.Comment(r,id='ewvs0c2'), praw.models.Comment(r,id='ewvrvt5'), praw.models.Comment(r,id='f0roa6b'), praw.models.Comment(r,id='f0rpmuf'), praw.models.Comment(r,id='f0rtgau'), praw.models.Comment(r,id='f0rl4zh'), praw.models.Comment(r,id='f0rfr82'), praw.models.Comment(r,id='f0rpbqc'), praw.models.Comment(r,id='epp3ghq'), praw.models.Comment(r,id='epp4qui'), praw.models.Comment(r,id='eppchkc'), praw.models.Comment(r,id='eaw8rtu'), praw.models.Comment(r,id='eaw55dc'), praw.models.Comment(r,id='eaw87m6'), praw.models.Comment(r,id='eaw3ryv'), praw.models.Comment(r,id='eavyr6w'), praw.models.Comment(r,id='eavr026'), praw.models.Comment(r,id='eaw343e'), praw.models.Comment(r,id='eavwcv7'), praw.models.Comment(r,id='eavq3jn'), praw.models.Comment(r,id='eavqt8p'), praw.models.Comment(r,id='eavpudd'), praw.models.Comment(r,id='eaw3jpe'), praw.models.Comment(r,id='eavujfz'), praw.models.Comment(r,id='eavygxm'), praw.models.Comment(r,id='eaw1068'), praw.models.Comment(r,id='eavoawd'), praw.models.Comment(r,id='eaw4rvf'), praw.models.Comment(r,id='eavti7c'), praw.models.Comment(r,id='eaw4ma2'), praw.models.Comment(r,id='eavz0bd'), praw.models.Comment(r,id='eavsxt9'), praw.models.Comment(r,id='eaw23fb'), praw.models.Comment(r,id='eavxo8o'), praw.models.Comment(r,id='eavt5dk'), praw.models.Comment(r,id='eavsjcy'), praw.models.Comment(r,id='eaw2sua'), praw.models.Comment(r,id='eavq211'), praw.models.Comment(r,id='eavtvk6'), praw.models.Comment(r,id='eavo438'), praw.models.Comment(r,id='eavptcq'), praw.models.Comment(r,id='eavqeqb'), praw.models.Comment(r,id='eavpd0e'), praw.models.Comment(r,id='eavzxar'), praw.models.Comment(r,id='eavpjy2'), praw.models.Comment(r,id='eemw86x'), praw.models.Comment(r,id='eemqzgx'), praw.models.Comment(r,id='eemps5e'), praw.models.Comment(r,id='eemstld'), praw.models.Comment(r,id='eemq0s9'), praw.models.Comment(r,id='ee34iwd'), praw.models.Comment(r,id='ee2xlic'), praw.models.Comment(r,id='ee2tb13'), praw.models.Comment(r,id='ee2r08e'), praw.models.Comment(r,id='ee2xswv'), praw.models.Comment(r,id='ee2lpbg'), praw.models.Comment(r,id='ee2n47n'), praw.models.Comment(r,id='ee2kwmg'), praw.models.Comment(r,id='ee2onrb'), praw.models.Comment(r,id='ee2mvlv'), praw.models.Comment(r,id='ee2k4lk'), praw.models.Comment(r,id='ee2n4mo'), praw.models.Comment(r,id='ee2qh6c'), praw.models.Comment(r,id='ee2k4c7'), praw.models.Comment(r,id='ee2uff0'), praw.models.Comment(r,id='ee2p7hg'), praw.models.Comment(r,id='ee2ogd0'), praw.models.Comment(r,id='ee2o0iu'), praw.models.Comment(r,id='ee2mx8m'), praw.models.Comment(r,id='ee2s9jj'), praw.models.Comment(r,id='ee35whf'), praw.models.Comment(r,id='ee2klhg'), praw.models.Comment(r,id='ee2q0cv'), praw.models.Comment(r,id='ee2oji9'), praw.models.Comment(r,id='ee2vtil'), praw.models.Comment(r,id='ee2m5o8'), praw.models.Comment(r,id='ee2orqo'), praw.models.Comment(r,id='ee2tir1'), praw.models.Comment(r,id='ee32izz'), praw.models.Comment(r,id='ee2wl6n'), praw.models.Comment(r,id='ee321fc'), praw.models.Comment(r,id='ee2nk8t'), praw.models.Comment(r,id='ee2w7sw'), praw.models.Comment(r,id='ee2xkbj'), praw.models.Comment(r,id='ee2pk6q'), praw.models.Comment(r,id='ee34rrm'), praw.models.Comment(r,id='ee2r1ve'), praw.models.Comment(r,id='ee2w46g'), praw.models.Comment(r,id='ee2tz95'), praw.models.Comment(r,id='ey8a1kt'), praw.models.Comment(r,id='ey8hrf0'), praw.models.Comment(r,id='ey8a8gi'), praw.models.Comment(r,id='ey8et8h'), praw.models.Comment(r,id='ey8e9g6'), praw.models.Comment(r,id='ey8f2eh'), praw.models.Comment(r,id='ey8bmrs'), praw.models.Comment(r,id='ey8dqhf'), praw.models.Comment(r,id='ey8bxxx'), praw.models.Comment(r,id='ey8dzau'), praw.models.Comment(r,id='er3t7py'), praw.models.Comment(r,id='ecgxnsl'), praw.models.Comment(r,id='ec4q6y8'), praw.models.Comment(r,id='ec4jakb'), praw.models.Comment(r,id='eed6ox6'), praw.models.Comment(r,id='eed2r5g'), praw.models.Comment(r,id='eed146n'), praw.models.Comment(r,id='eed4008'), praw.models.Comment(r,id='eidu3es'), praw.models.Comment(r,id='eidwb0h'), praw.models.Comment(r,id='eidt3zm'), praw.models.Comment(r,id='eidwury'), praw.models.Comment(r,id='eidsjgn'), praw.models.Comment(r,id='eiduay3'), praw.models.Comment(r,id='eidsddn'), praw.models.Comment(r,id='dzj4aj0'), praw.models.Comment(r,id='dzj3hlw'), praw.models.Comment(r,id='dzj376e'), praw.models.Comment(r,id='dzj2ua4'), praw.models.Comment(r,id='ejnyhbf'), praw.models.Comment(r,id='eqamyqf'), praw.models.Comment(r,id='eqaorac'), praw.models.Comment(r,id='ehcjpf8'), praw.models.Comment(r,id='ehcrbsh'), praw.models.Comment(r,id='ehcr6b1'), praw.models.Comment(r,id='ee4nih2'), praw.models.Comment(r,id='ee5d4j0'), praw.models.Comment(r,id='drqkxmo'), praw.models.Comment(r,id='ec7hpjw'), praw.models.Comment(r,id='e8j0scf'), praw.models.Comment(r,id='do8bwex'), praw.models.Comment(r,id='e1bg4zo'), praw.models.Comment(r,id='dfw81tq'), praw.models.Comment(r,id='dfw0qhn'), praw.models.Comment(r,id='dppxuiy'), praw.models.Comment(r,id='ekkt6hc'), praw.models.Comment(r,id='ekkpyd7'), praw.models.Comment(r,id='ekkqmgu'), praw.models.Comment(r,id='ekkwo5f'), praw.models.Comment(r,id='ekmkebc'), praw.models.Comment(r,id='ekmvnhb'), praw.models.Comment(r,id='ekmof92'), praw.models.Comment(r,id='ekm473i'), praw.models.Comment(r,id='ec91wqf'), praw.models.Comment(r,id='ec8xdj5'), praw.models.Comment(r,id='ec8z6zq'), praw.models.Comment(r,id='ec9c0rm'), praw.models.Comment(r,id='eg12koe'), praw.models.Comment(r,id='eg0ynnt'), praw.models.Comment(r,id='ehyuwv8'), praw.models.Comment(r,id='ehyvx95'), praw.models.Comment(r,id='ehyvdlc'), praw.models.Comment(r,id='ehyvkfa'), praw.models.Comment(r,id='ehypggt'), praw.models.Comment(r,id='efcqlk0'), praw.models.Comment(r,id='d9s9rfq'), praw.models.Comment(r,id='d9s9o47'), praw.models.Comment(r,id='d9s9uuv'), praw.models.Comment(r,id='d9sancm'), praw.models.Comment(r,id='d9s9j1h'), praw.models.Comment(r,id='d9s7reh'), praw.models.Comment(r,id='d9s7qmt'), praw.models.Comment(r,id='e01lm8d'), praw.models.Comment(r,id='f3bq2qx'), praw.models.Comment(r,id='f3ct2b4'), praw.models.Comment(r,id='f3cq0ig'), praw.models.Comment(r,id='f3kppri'), praw.models.Comment(r,id='f3lmrco'), praw.models.Comment(r,id='f3lqd4b'), praw.models.Comment(r,id='f3lun2n'), praw.models.Comment(r,id='f3mbc2v'), praw.models.Comment(r,id='dl4rh3d'), praw.models.Comment(r,id='e7ruem2'), praw.models.Comment(r,id='doi3q16'), praw.models.Comment(r,id='doidhhn'), praw.models.Comment(r,id='dlqhic5'), praw.models.Comment(r,id='dlqcxeu'), praw.models.Comment(r,id='dlqcy76'), praw.models.Comment(r,id='esl156r'), praw.models.Comment(r,id='eskx8u3'), praw.models.Comment(r,id='esluwhu'), praw.models.Comment(r,id='dyj35y9'), praw.models.Comment(r,id='dyiyaha'), praw.models.Comment(r,id='dmswnzd'), praw.models.Comment(r,id='f3plrlp'), praw.models.Comment(r,id='f3pckf3'), praw.models.Comment(r,id='e3o0rml'), praw.models.Comment(r,id='e3odbfj'), praw.models.Comment(r,id='e3o5nra'), praw.models.Comment(r,id='em0gxs1'), praw.models.Comment(r,id='eta7lwd'), praw.models.Comment(r,id='dgsm4vp'), praw.models.Comment(r,id='dgshsb4'), praw.models.Comment(r,id='dl0k41z'), praw.models.Comment(r,id='dl0mxv3'), praw.models.Comment(r,id='dl0km9y'), praw.models.Comment(r,id='dl0ltu7'), praw.models.Comment(r,id='e2p4sg5'), praw.models.Comment(r,id='f39w1qo'), praw.models.Comment(r,id='eickazv'), praw.models.Comment(r,id='ekjrg7a'), praw.models.Comment(r,id='ekjmm2e'), praw.models.Comment(r,id='ekjmuok'), praw.models.Comment(r,id='ehszih9'), praw.models.Comment(r,id='ehsqx30'), praw.models.Comment(r,id='ehu4tl5'), praw.models.Comment(r,id='dkrrtiw'), praw.models.Comment(r,id='euysy5b'), praw.models.Comment(r,id='euytraw'), praw.models.Comment(r,id='ealnpgr'), praw.models.Comment(r,id='doyruyb'), praw.models.Comment(r,id='doyrp3u'), praw.models.Comment(r,id='e67roxo'), praw.models.Comment(r,id='es01hjc'), praw.models.Comment(r,id='erzwx9o'), praw.models.Comment(r,id='f2jpogi'), praw.models.Comment(r,id='f2kb43t'), praw.models.Comment(r,id='f2jz96w'), praw.models.Comment(r,id='dq7nwxz'), praw.models.Comment(r,id='dvt3xiw'), praw.models.Comment(r,id='csd0n1e'), praw.models.Comment(r,id='dv9kjhn'), praw.models.Comment(r,id='dvydoq4'), praw.models.Comment(r,id='dvyedgo'), praw.models.Comment(r,id='dpdxgtj'), praw.models.Comment(r,id='ed3xzi0'), praw.models.Comment(r,id='f1rgpw3'), praw.models.Comment(r,id='f1qh2f6'), praw.models.Comment(r,id='f1q6uza'), praw.models.Comment(r,id='f1qwhux'), praw.models.Comment(r,id='ew24hmy'), praw.models.Comment(r,id='ew25aja'), praw.models.Comment(r,id='ew1ys7u'), praw.models.Comment(r,id='e09zlbe'), praw.models.Comment(r,id='e0a0lo8'), praw.models.Comment(r,id='egi96mp'), praw.models.Comment(r,id='erhzjlh'), praw.models.Comment(r,id='eri22br'), praw.models.Comment(r,id='ec38dzy'), praw.models.Comment(r,id='ec38n0s'), praw.models.Comment(r,id='ec3dlyi'), praw.models.Comment(r,id='ec35hmr'), praw.models.Comment(r,id='edf6wnr'), praw.models.Comment(r,id='edetvff'), praw.models.Comment(r,id='edevmys'), praw.models.Comment(r,id='edey43g'), praw.models.Comment(r,id='edex5f1'), praw.models.Comment(r,id='edfgrzt'), praw.models.Comment(r,id='esqknzo'), praw.models.Comment(r,id='esq623a'), praw.models.Comment(r,id='esq791s'), praw.models.Comment(r,id='esqfxl0'), praw.models.Comment(r,id='esq92ox'), praw.models.Comment(r,id='eyrxowi'), praw.models.Comment(r,id='eys0nqm'), praw.models.Comment(r,id='ey928qv'), praw.models.Comment(r,id='ejo9bqy'), praw.models.Comment(r,id='ejo9d9g'), praw.models.Comment(r,id='ejop1qx'), praw.models.Comment(r,id='ecoxn7y'), praw.models.Comment(r,id='ezg6060'), praw.models.Comment(r,id='ezg6nm5'), praw.models.Comment(r,id='ef9lvmu'), praw.models.Comment(r,id='ef9ug70'), praw.models.Comment(r,id='efa2jl5'), praw.models.Comment(r,id='eblhl3a'), praw.models.Comment(r,id='eblnuhn'), praw.models.Comment(r,id='e6eycan'), praw.models.Comment(r,id='e6ex0fz'), praw.models.Comment(r,id='ed5sotc'), praw.models.Comment(r,id='ed646xi'), praw.models.Comment(r,id='ehq7xqe'), praw.models.Comment(r,id='ehq1f7c'), praw.models.Comment(r,id='ehq4p4d'), praw.models.Comment(r,id='ehq8qlk'), praw.models.Comment(r,id='drzcfvu'), praw.models.Comment(r,id='e2c2rtu'), praw.models.Comment(r,id='eiejl4u'), praw.models.Comment(r,id='ee2m7wn'), praw.models.Comment(r,id='dmg1dqt'), praw.models.Comment(r,id='dmglel8'), praw.models.Comment(r,id='ev3khxw'), praw.models.Comment(r,id='ev3h8pa'), praw.models.Comment(r,id='ev3kk54'), praw.models.Comment(r,id='eb1dva6'), praw.models.Comment(r,id='eb1kibo'), praw.models.Comment(r,id='ei9eqzj'), praw.models.Comment(r,id='ed1rog5'), praw.models.Comment(r,id='droo0a8'), praw.models.Comment(r,id='dwhbd53'), praw.models.Comment(r,id='dwh80mb'), praw.models.Comment(r,id='eanaq8z'), praw.models.Comment(r,id='ee4hk39'), praw.models.Comment(r,id='dhv9dy4'), praw.models.Comment(r,id='drsv0aa'), praw.models.Comment(r,id='emwcnd2'), praw.models.Comment(r,id='emwcxhq'), praw.models.Comment(r,id='emwhhs6'), praw.models.Comment(r,id='emwq4sq'), praw.models.Comment(r,id='dboqa3w'), praw.models.Comment(r,id='dbop4vt'), praw.models.Comment(r,id='dbonl3d'), praw.models.Comment(r,id='dbonsz9'), praw.models.Comment(r,id='dboors0'), praw.models.Comment(r,id='ex31gjt'), praw.models.Comment(r,id='eii4wig'), praw.models.Comment(r,id='eii5899'), praw.models.Comment(r,id='eii5lvx'), praw.models.Comment(r,id='eii51pi'), praw.models.Comment(r,id='eii4ymh'), praw.models.Comment(r,id='dkb4qqx'), praw.models.Comment(r,id='e2uc29x'), praw.models.Comment(r,id='ek0qg02'), praw.models.Comment(r,id='ek0ptpk'), praw.models.Comment(r,id='eq3usua'), praw.models.Comment(r,id='d71lioa'), praw.models.Comment(r,id='d71r5v4'), praw.models.Comment(r,id='d71lqp5'), praw.models.Comment(r,id='cjnjn9z'), praw.models.Comment(r,id='cjnjqz2'), praw.models.Comment(r,id='cjnke37'), praw.models.Comment(r,id='cjnkc5m'), praw.models.Comment(r,id='cjnl7j9'), praw.models.Comment(r,id='cjnjcfe'), praw.models.Comment(r,id='cjnjh3i'), praw.models.Comment(r,id='cjnjtuh'), praw.models.Comment(r,id='cjnjhel'), praw.models.Comment(r,id='cjnjrw6'), praw.models.Comment(r,id='cjnjai1'), praw.models.Comment(r,id='cjnjl00')]
from praw.models import MoreComments



for i in range(9):

    for j in random.sample(range(0, 100), 25):  # the second dimension is the post. Notice here is the random sampling

        comments = posts[0][j].comments

        for comment in comments:

            if isinstance(comment, MoreComments):   # avoid comment that are not top-level

                continue

            try:

                keys = list(comment.gildings.keys())   # list of awards the comment has, if any

            except AttributeError:

                pass

            if bool('gid_1' in keys) ^ bool('gid_2' in keys) ^ bool('gid_3' in keys):

                print(comment.gildings)

                worthy_comments.append(comment)
from IPython.display import clear_output



all_comments = []



for i in range(len(worthy_comments)):

    keys = list(worthy_comments[i].gildings.keys())

    silvers = 0

    golds = 0

    platinums = 0

    if bool('gid_1' in keys):

        silvers = worthy_comments[i].gildings['gid_1']

    if bool('gid_2' in keys):

        golds = worthy_comments[i].gildings['gid_2']

    if bool('gid_3' in keys):

        platinums = worthy_comments[i].gildings['gid_3']  

    

    s = praw.models.Submission(r, id=worthy_comments[i].link_id[3:])   #this makes the submission object non-lazy, so we can access the time the submission was made

    _ = s.title

    entry = [

        worthy_comments[i].subreddit.display_name, 

        silvers, 

        golds, 

        platinums, 

        worthy_comments[i].created_utc,

        s.created_utc,

        worthy_comments[i].created_utc - s.created_utc,

        worthy_comments[i].link_id[3:],

        worthy_comments[i].id,

        worthy_comments[i].permalink

    ]

    all_comments.append(entry)

    

    # jsut to show progress for my sanity

    clear_output()

    print('comment ' + str(i) + '/' + str(len(worthy_comments) - 1) + '   :   ' + str("{:.1f}".format((i / len(worthy_comments)) * 100)) + '%')

    

    

df = pd.DataFrame(all_comments, columns=

                  ['Subreddit',

                   'Silver',

                   'Gold',

                   'Platinum',

                   'Comment Time',

                   'Post Time',

                   'Time Difference',

                   'PostID',

                   'CommentID',

                   'permalink'])

    

df
import os

os.chdir(r'/kaggle/working')



df.to_csv(r'df_top_comments_with_awards.csv')        #This gives you a download link to save the DataFrame you generated



from IPython.display import FileLink

FileLink(r'df_top_comments_with_awards.csv')
df = pd.read_csv(r'../input/top-comments-with-awards/df_top_comments_with_awards.csv')
df['Time Difference'].describe()
import matplotlib

import matplotlib.pyplot as plt

from matplotlib import colors

%matplotlib inline



matplotlib.rcParams.update({'font.size': 15})



def createFigure(x, title):

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.set_facecolor('xkcd:white')

    # formatting

    ax.set_title(title)

    ax.set_ylabel('# of comments')

    ax.set_xlabel('Hours after parent post')



    # Color each bar by height

    N, bins, patches = ax.hist(x, bins=range(int(min(y)), int(max(y)) + 1, 1), edgecolor='gray')

    fracs = N / N.max()

    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):

        color = plt.cm.viridis(norm(thisfrac))

        thispatch.set_facecolor(color)

    

    plt.show()



xAll = df['Time Difference'] /3600

createFigure(xAll, 'Frequency distribution of top-level awarded comments \n depending on how long the comment was made after the parent post')
xSilver = df[(df['Silver'] > 0) & (df["Gold"] == 0) & (df["Platinum"] == 0)]['Time Difference'] / 3600

xGold = df[(df['Silver'] == 0) & (df["Gold"] > 0) & (df["Platinum"] == 0)]['Time Difference'] / 3600

xPlatinum = df[(df['Silver'] == 0) & (df["Gold"] == 0) & (df["Platinum"] > 0)]['Time Difference'] / 3600



createFigure(xSilver, 'Frequency distribution of silver-only comments \n with respect to time after parent post')

createFigure(xGold,'Frequency distribution of gold-only comments \n with respect to time after parent post')

createFigure(xPlatinum,'Frequency distribution of platinum-only comments \n with respect to time after parent post')