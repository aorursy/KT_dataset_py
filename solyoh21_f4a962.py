
{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 14966,
      "digest": "sha256:0bb414c1080d09555e8827e2ba01e8e4094e6688f560c5ed78d66490c5d93102"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 50382957,
         "digest": "sha256:7e2b2a5af8f65687add6d864d5841067e23bd435eb1a051be6fe1ea2384946b4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 222909892,
         "digest": "sha256:59c89b5f9b0c6d94c77d4c3a42986d420aaa7575ac65fcd2c3f5968b3726abfc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 195204532,
         "digest": "sha256:4017849f9f85133e68a4125e9679775f8e46a17dcdb8c2a52bbe72d0198f5e68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1522,
         "digest": "sha256:c8b29d62979a416da925e526364a332b13f8d5f43804ae98964de2a60d47c17a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 717,
         "digest": "sha256:12004028a6a740ac35e69f489093b860968cc37b9668f65b1e2f61fd4c4ad25c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 247,
         "digest": "sha256:3f09b9a53dfb03fd34e35d43694c2d38656f7431efce0e6647c47efb5f7b3137"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 408,
         "digest": "sha256:03ed58116b0cb733cc552dc89ef5ea122b6c5cf39ec467f6ad671dc0ba35db0c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 331594702,
         "digest": "sha256:7844554d9ef75bb3f1d224e166ed12561e78add339448c52a8e5679943b229f1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 112943346,
         "digest": "sha256:d02cf9213088950ea80cc37d21eed912de9a9b64170c046173c2de5c9b5eedc9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 425,
         "digest": "sha256:b89ff65d69ce89fe9d05fe3acf9f89046a19eaed148e80a6e167b93e6dc26423"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 5476,
         "digest": "sha256:d7a15e9b63f265b3f895e4c9f02533d105d9b277e411b93e81bb98972018d11a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1949,
         "digest": "sha256:f2b5041ea52a4ce4a0e0c7665327b5e01fe915c6fd78ccd5b722f2fb99e24da5"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2488478912,
         "digest": "sha256:05e6d738c6522686b0fa5851459a77fe2527f4b49cb0a54c1e3b200235f8439e"
      }
   ]
}

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