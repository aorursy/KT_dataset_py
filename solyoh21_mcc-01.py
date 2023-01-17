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

      "size": 33355,

      "digest": "sha256:7a5772bb31c2ac957e648cdf6c2df60dd0f1319ab820a462b619ee7425daaa20"

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

         "size": 527,

         "digest": "sha256:3629d07017dfd912716122fbb885d837d26288d06022c2db4b15a8b10825f3af"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 457,

         "digest": "sha256:af3c21fe2b3c51a257748946231568e792d8bbf0d2439550c8752762012dc526"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 21088795,

         "digest": "sha256:fb85b7a23e3ca2ebbd96564122d3641eb2377aefef1790449b613e9a178d1713"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 109595514,

         "digest": "sha256:e29c6c5047383db8628c70f97a19f3cead17f7a916bbf09ae5ccae13732b7ce8"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 277951818,

         "digest": "sha256:263839d8733ea7eb00f296e5b10d0d850275c95197bbc744f87b6f91f2444b0c"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 207937796,

         "digest": "sha256:6d2131fe5cdc61c560758eb5981f258c1798490f726e6193e7e0e8ac533a8296"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 96474633,

         "digest": "sha256:dd35656edd3dbaa072376eb72a69b8b87e12a4f3b973c29a684afeb17e2926b3"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 108017763,

         "digest": "sha256:f2ec0cf92cc74c0d5925f0d82d09b2475b4ae809fdce7d0529d89f59f63feafa"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1026760725,

         "digest": "sha256:40778247fa269c6c238f046dc7f4a0f6229bef752d1349b15ce0c7bbbec082a3"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 74960047,

         "digest": "sha256:6484cf54a8fa926ef31ec1ded103f6ec77f27facb3b41f928dbbf9f48d200852"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 46045702,

         "digest": "sha256:75ed4ecf894d1cd5446a75e912bd56558ea64130ea9ae2f506786d2f4fa4e1a5"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 59099198,

         "digest": "sha256:99c1b7275c7542950bba439124c4b3d757777d3b74607c2aa030988f756c3aa4"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 394268068,

         "digest": "sha256:f1bcb0101d1f558958a75c982e4828dde7c1cfa8bc60b8e7443cec80c6efc410"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 46393926,

         "digest": "sha256:99ae79e136a288018f3dfaba295ac9852d300906d9e2df8b8c49448adf421e4a"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 105262909,

         "digest": "sha256:ca69599994e335742556ca9e2344b03eee1ee37c20819c43e8310f6f4f1b7632"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 890007955,

         "digest": "sha256:d553244461d0da96411cd7220c5b3c6f3734f30c4600a61f054676dee4b256dd"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 179773681,

         "digest": "sha256:e55e0789c408f05f631285e294ffa1918f193055cd0179794bf5088bf56e410d"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 10624139,

         "digest": "sha256:fdd16d19d11a0ee65ef2804afb46cddbd72657beec4186997e2e7e2c952027f6"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1845114,

         "digest": "sha256:51ce7cbebc05b188ccf21ccb0d7c851cb0aa9858a97b1cff24fffa777c05e2e4"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 79286588,

         "digest": "sha256:6f61f8e2053618cf4a14d1d8a0ad94f8f5a8667dbff02cc8d8cb55da28851186"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 3264,

         "digest": "sha256:24d6c2fc6f23c36fe025d2b644587e44bc2e7832d059f842810b954cdb5b2457"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 2165,

         "digest": "sha256:d185a954fdb6e50bbee5ca2ada347b149039ce99fbf506aba998e7f2f67c1a73"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1269,

         "digest": "sha256:69ec9e29cfe6653727358fa0c94dd77aa3ab5f1dadf7cc8955f8b1a929acd603"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 646,

         "digest": "sha256:5813a8e61222b561bdc0d9a6dec27baa0d11c3ce4afb50586212a19f4dbe7d5a"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 2052,

         "digest": "sha256:aefae160eb4a9edc4d39a71bf69a3751fc64a4509222a0030445cb43730f1df8"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 878,

         "digest": "sha256:4af50d7cad7f7889df18a8a4c6a747b1dd7f46f5d16c3956e0efd6ceb95163fc"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 378,

         "digest": "sha256:ba7885708ce113cb634b55a1eebd9b014281e974b5fb8b053f13e2d2d8976a9b"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 211,

         "digest": "sha256:2b69da9c605bbac66b4a537de22be25e23df886375882448ea15043e071938b3"

      }

   ]

}

{

  "cells": [

    {

      "metadata": {

        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",

        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",

        "trusted": true

      },

      "cell_type": "code",

      "source": "%load_ext tensorboard.notebook\n%tensorboard --logdir /logs",

      "execution_count": null,

      "outputs": []

    },

    {

      "metadata": {

        "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",

        "collapsed": true,

        "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",

        "trusted": false

      },

      "cell_type": "code",

      "source": "",

      "execution_count": null,

      "outputs": []

    }

  ],

  "metadata": {

    "kernelspec": {

      "display_name": "Python 3",

      "language": "python",

      "name": "python3"

    },

    "language_info": {

      "name": "python",

      "version": "3.6.4",

      "mimetype": "text/x-python",

      "codemirror_mode": { "name": "ipython", "version": 3 },

      "pygments_lexer": "ipython3",

      "nbconvert_exporter": "python",

      "file_extension": ".py"

    }

  },

  "nbformat": 4,

  "nbformat_minor": 1

}
