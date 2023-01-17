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

{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 33425,
      "digest": "sha256:d86dac89e1c53e72dff469bc8c8d91bdca684618ae5e4fff1b9989062a429ee5"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 26692096,
         "digest": "sha256:423ae2b273f4c17ceee9e8482fa8d071d90c7d052ae208e1fe4963fceb3d6954"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 35365,
         "digest": "sha256:de83a2304fa1f7c4a13708a0d15b9704f5945c2be5cbb2b3ed9b2ccb718d0b3d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 852,
         "digest": "sha256:f9a83bce3af0648efaa60b9bb28225b09136d2d35d0bed25ac764297076dec1b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 163,
         "digest": "sha256:b6b53be908de2c0c78070fff0a9f04835211b3156c4e73785747af365e71a0d7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 403170736,
         "digest": "sha256:5650063cfbfb957d6cfca383efa7ad6618337abcd6d99b247d546f94e2ffb7a9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 81117097,
         "digest": "sha256:89142850430d0d812f21f8bfef65dcfb42efe2cd2f265b46b73f41fa65bef2fe"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 6868,
         "digest": "sha256:498b10157bcd37c3d4d641c370263e7cf0face8df82130ac1185ef6b2f532470"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 144376365,
         "digest": "sha256:a77a3b1caf74cc7c9fb700cab353313f1b95db5299642f82e56597accb419d7c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1551901872,
         "digest": "sha256:0603289dda032b5119a43618c40948658a13e954f7fd7839c42f78fd0a2b9e44"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 467065,
         "digest": "sha256:c3ae245b40c1493b89caa2f5e444da5c0b6f225753c09ddc092252bf58e84264"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 324,
         "digest": "sha256:67e85692af8b802b6110c0a039f582f07db8ac6efc23227e54481f690f1afaae"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 450,
         "digest": "sha256:ea72ab3b716788097885d2d537d1d17c9dc6d9911e01699389fa8c9aa6cac861"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 197,
         "digest": "sha256:b02850f0d90ca01b50bbfb779bcf368507c266fc10cc1feeac87c926e9dda2c1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 198,
         "digest": "sha256:4295de6959cedecdd0ba31406e15c19e38c13c0ebc38f3d6385725501063ef46"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:d651a7c122d62d2869af2a5330c756f2f4b35a8e44902174be5c8ce1ad105edd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:69e0b993e5f56695ee76b3776275dac236d38d32ba1f380fd78b900232e006ec"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 526,
         "digest": "sha256:41129095d013d12e9ad82ee57f03ba8c2db2ea95121551d101c9c746a6c99cf9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 458,
         "digest": "sha256:a8dfbfc20da6685bdbcc74b911f5e48ef29d066fc2aa011773a00782d943a213"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21089025,
         "digest": "sha256:c824d50966cb902e5156a7edfd937e31e335db9f3c45dc1b14d42c2e603c8dbc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 109595597,
         "digest": "sha256:11232601f797c7a06b8fa4ac42ebd1eca4a9ae6bfc772cbef6a20ed67c163315"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 278720756,
         "digest": "sha256:b1c2034aae3d90c20b4a78b154332294de7187e9f845c71f93a7ba1708a5fcc6"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 207937621,
         "digest": "sha256:94bb4742f64c45c3666e60036f0d964f7946666575b0dabde266dcd77cc854da"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 96474602,
         "digest": "sha256:ec963581389fa9a74ee89beb418b11b9d660c25fe294a38bf2503947a8971a8a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 108017816,
         "digest": "sha256:55d60fcd484b141be38db530f94dde3674f4d7e0ebfb1c6aa20e29627eb8c054"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1026759978,
         "digest": "sha256:b0c5ff5de7a357b4a6b825fd78bf2ffdf3216f8b8ad5f189764f71766509c581"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 74960096,
         "digest": "sha256:ce8b8b22da4129d929c6f2a7fde8a55576b0b6ba9446fe75c023aab6f401456e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46047484,
         "digest": "sha256:873ee619235e93369ed7bd9071e9b22df7871f85c0891982e36425db1b14a4c4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 59099746,
         "digest": "sha256:50c4c61abf8dfb38ee73933a3d9b9ca1c197422c7ff3156ebd899930dd440dc7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 394269423,
         "digest": "sha256:c28a8d773229baf96dd4f86d8c6bc64b3822babfabbbda7991037766505a705c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46393689,
         "digest": "sha256:8046c894297ee49f4ac83282ab0fe3b4edb23bb78c7e5c0432b4c3b4d01c8f3e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 105258382,
         "digest": "sha256:a519cbc32d7d66b8e6b3dc734c6e736377e4a2d36a2537d8cd1a461ad5f6af01"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 890007190,
         "digest": "sha256:1e95d5f198702a863271fb4f1ba2f1e76491e303c4c65b862679c63fdad5a4dd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 179775031,
         "digest": "sha256:f1768ffef273be6b2c52734cf9562c778297ddeef1a5d5876fd4de893730297a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 10624155,
         "digest": "sha256:61d97647a885eacfa7793fb907e717f95cd80c145805cf9fde4c5111ca77cce1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1845209,
         "digest": "sha256:82bb599464ad064cc3d3d73b09dff06ddbb0abbe1c4ba8596ac0bf307a28deea"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 79288497,
         "digest": "sha256:44cbc22308040bfede5bc20ac4faf2e14147bbb3997b1f3f384b8090c00e8a19"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3266,
         "digest": "sha256:fc367bdd185360c07698ee100b14a7ee86b3760905c2bb057529ce940e85958f"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2167,
         "digest": "sha256:f5d58949fdd76de68e83222f80f6cb451527dff7d8711631384d54b4b8872aca"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1270,
         "digest": "sha256:3a2e2b2b7171d815e5383d81f72cb9b20f0313b2c872730ae3bbb09441e87232"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 643,
         "digest": "sha256:fa58daee3d7a2cda3f88eb80c49da2f7118b990a85f4c6d33f82eb9e4f8558ca"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2052,
         "digest": "sha256:0e4e44772c975df8e7d0e9f0edd5a91d2ff5ed4df448b5fe4195510e1818c1f4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 876,
         "digest": "sha256:d1764c90f2cf94e1b5cc145463f10e1f618ad2ee465e1d11def411ddf69e1456"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 383,
         "digest": "sha256:5715055abfdce5503b809d7b14225cc81c948a20633fc8375347b816dd83d5c8"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 211,
         "digest": "sha256:68eace16eba2a633e07bb53ea03766033034a164551c8e71035b5b15d6178c70"
      }
   ]
}
