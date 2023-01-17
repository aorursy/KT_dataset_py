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

      "size": 33353,

      "digest": "sha256:fd13cb6386677dc7ace2ca8a471f64d88e9b44fdb97b9b544803e04b028a2af9"

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

         "digest": "sha256:3f6923639a16c82f272e53e616792a0eab336543492dc1806121a026838af933"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 457,

         "digest": "sha256:2e35b30c8b5a638d926596825ddcf130f2604139f15238971b440cb21f02b372"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 21089133,

         "digest": "sha256:bddf9e7b07fd851fd2548cd8711f0d5b5ae1750bdebcde7b37aa11795278a89e"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 109595462,

         "digest": "sha256:822afcd5ece6b7316c4c079edfaa01bed90a522824ef7b827e60f47052a204d5"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 278055204,

         "digest": "sha256:c55a8e320009ace89dd4bfd13c12f13f4da0852c0826e0146860c0929b39a08c"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 207937054,

         "digest": "sha256:7d17ec2d3bacf3194b5610f3276c5ea4308ea64da9df40d7f7840e2fcd3fed83"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 96474602,

         "digest": "sha256:4ca36c52231e4c03623c8824600e395805ba058b5f1f50a5da547f4581b21a06"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 108017458,

         "digest": "sha256:947748d5ae5cfe16733db29a5cb72a271c0cd924cb2fa87efcc8e9a39128eb1b"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1026761485,

         "digest": "sha256:3a0fd3cb9609393e6a9080e780a80f398b0d4579b6ab2b8fc08e40405d0a7540"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 74960141,

         "digest": "sha256:ce3827062a70cd5258069ead98c1fdc0cfec72726ea240e273405af9f2cafc04"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 46047390,

         "digest": "sha256:7058e0242d55c986bbfe5929989d2eb140ea3238df676f39afae85e6115bff85"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 59099645,

         "digest": "sha256:0ef44ae4d4a0c2988aa50d64e7b31cc7e0c977519dc4167a793f1e9ff08eaa98"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 394266269,

         "digest": "sha256:c59a9c0687e2db7c8c9a2392787329c020a23ec636b2ed3b211eaa47e1355447"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 46394201,

         "digest": "sha256:5c82552656d5c24234fc3a7c7b49e52b3a11aee279b14de74ea189981f7070d0"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 105263829,

         "digest": "sha256:4e0d25bbfff5314a9f74ea9cc035704d78daa0444c1ce183530c76212c79d196"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 890007199,

         "digest": "sha256:f460063bea9df9900d676cb293e44d99b6d5c80f2748cacc9d9529172651319f"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 179777604,

         "digest": "sha256:13ef43acf95cf8003c23943928100f2e4264fd51616adf2fae53631f28538a6f"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 10624178,

         "digest": "sha256:5dd48754a5389341b9449dbbfe670b49d2a3527f1d1ffd3a8bdaf29d3020cf45"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1845146,

         "digest": "sha256:95556e3c4834e1b6bb198e51826260d67070a98475caf6c6eb72a3f377ac8cc1"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 79288151,

         "digest": "sha256:a259af45c4c8b21f22db3fe6cf9d2b72c8776c51fc71ce2ed3cf5c4fba7d7a36"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 3262,

         "digest": "sha256:21fbe1b95cef1fc4de465f9a16d59db2455d3478da3894cd2c1f0795fff674c9"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 2168,

         "digest": "sha256:e73d46e654c4f81425a80114bf2e334e64658619f22382f074cf56a163664701"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1272,

         "digest": "sha256:4e4b9382ddf9413ffb5ccf1244bd03ce7d7e408b732146e2c9fb171b925852ea"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 645,

         "digest": "sha256:ca23ce1b8aa912ce61d9f9029f906c0cd1ae42849cafac2a9041685f25299dbe"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 2052,

         "digest": "sha256:60f3926aaf3f2d7d94242fa7d52c8f79341750afbb35cfe9da897605e039deb4"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 877,

         "digest": "sha256:036bdfd6ebb40ec9472a42bb13cfcc9d2d8b8c42bd41987b196a88dae8331a0a"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 382,

         "digest": "sha256:8442f6c099ace12b067c330b0a69fe520a57b16396fc1cf6cb8f5e91650afedd"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 213,

         "digest": "sha256:a7f74a50c8ea1d4eec6e3baef5d01bde3a243d8ad5bc2bb8e077dd7772724e92"

      }

   ]

}
