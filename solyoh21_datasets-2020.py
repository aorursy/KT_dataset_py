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


{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 33524,
      "digest": "sha256:df06a663adbbc19e20ee49cf1a3acfe4bffb34ddb8fe12b69f1a932210968321"
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
         "size": 525,
         "digest": "sha256:4bdee63a03eef498c72af47b237e77f79fa084af3b3dcc453b0abf5670ada4f2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 462,
         "digest": "sha256:6e001c98e0fb9b8a9bff310166ab7a57c87098fd386593e5751dc16b95920d0a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21088571,
         "digest": "sha256:b993fb3f0d801a4ab4739e328e5993b7d767ee197ab13fecef34f0f81f8ea736"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 109594846,
         "digest": "sha256:fe552aa20a1909859246a4269c13aaf0b830e02d54806aee357417d09fd145e6"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 278178438,
         "digest": "sha256:7f80e47aac034f1b24558546043f4679a9942db321546c1c7d544c0e8e2c9a5f"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 207938086,
         "digest": "sha256:dc8d3a5241b29b1e4f1b2f9267b90026f16bcf6f4d32cd52e309cc1401a3aa5a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 96474599,
         "digest": "sha256:7732a804278896b7be704a4934b65765f0e445943092e1f0ad2385019be22572"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 108018206,
         "digest": "sha256:a2cebabbc4c1d886f99ed03aac135e8dd28bbf8569b6012d085f51e937ce0100"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1026769385,
         "digest": "sha256:1444b7718af5743a628cfd122505171024f1329c1e2b8316e9fb7b5fdc038ba8"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 74960447,
         "digest": "sha256:46d6e310fbda0ce866c349be9beb6287ba22ea5ad494a4050c5136b2454c6c4f"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46046586,
         "digest": "sha256:05e67c24c2cb20c30fa5dc1c655fb1a38a7ec45ab289981511682a6b90d34fb5"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 59099727,
         "digest": "sha256:77d1d19c278fda673ba13be8869659a992a48dcc22c548fff06a3f74d2f6fb97"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 394269569,
         "digest": "sha256:e87bb90cf3c2ea94b24c35397bf041b128a74febc353e78063bb719536c2eb7e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46394431,
         "digest": "sha256:d1dcab3721550c1cf30db5ee1b1894f5357d44690b7a6ec345a918406dce1c3a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 105260890,
         "digest": "sha256:cda2ee9100ec5bf4e8afbca3bbc104777d00a39bbb3a104ac1cd17a58c06ee1a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 890007033,
         "digest": "sha256:dde0d51402c9f36d712dd8a528dd0fc3f0750213deeee3b38934eaade05c4c13"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 179779992,
         "digest": "sha256:876aeac3e66482c21d4f9d6290ed947f5ab91f1469aa7d4de6521a0925fc7265"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 10624034,
         "digest": "sha256:4431dd0a8e5a7b66eb0cfe28b41795e4fc8a585280dcf1cb6a6fb6c08e1aaf4c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1845085,
         "digest": "sha256:e3b72ecfcacff8732cf289aa842a29fa6bdce4f1ed0f8ef0219c9bd07e46eb16"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 79288180,
         "digest": "sha256:27b2623138f5b73c968e7eb53999d69bdd7306b8c61d68aed03a380a01d4d1fa"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3266,
         "digest": "sha256:76b66d86947e1c16c594f4364abb29843d869c1838ee86ac104b1c1b5c4c2087"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2164,
         "digest": "sha256:b39b99c4778515af636c383a58247efafd8fc82fd3b4aeeebefef6ba7447aaad"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1267,
         "digest": "sha256:8db11b1a3a89b5a8e09ba9f9332d8d0b55807ab8336084290a8106a8725f69df"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 643,
         "digest": "sha256:0bb1cc9a23a2f6cf79361168e9400a30e7d945b4bc481053a679b422379d8559"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2052,
         "digest": "sha256:0c83df332e6b116552faddc3f67d5ab5058fc0d55137071fd32dc67f0cfb44aa"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 877,
         "digest": "sha256:ed76be9c1d993fbad9c1d6c589c8db5186d3840bee3e38e8843d05232a476491"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 380,
         "digest": "sha256:d60d39a618881125eeb71355f7a46936ae567bc8b0717fb63dbe862e3d74d7ec"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:adb534500227d312aaab9c9fb804e9c4f782e2502a9480a301217517491796d2"
      }
   ]
}
