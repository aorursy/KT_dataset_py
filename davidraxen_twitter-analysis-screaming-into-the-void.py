import json



import string

import re



#Data visualization

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sns





#Date

import datetime as dt

import time



from math import pi

import os



import pandas as pd

import numpy as np



from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import warnings 

warnings.filterwarnings('ignore')
tweets = r'''

[ {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "user_mentions" : [ ],

    "urls" : [ ],

    "symbols" : [ ],

    "media" : [ {

      "expanded_url" : "https://twitter.com/davidraxen/status/1140952800697995266/photo/1",

      "indices" : [ "77", "100" ],

      "url" : "https://t.co/GToDd8hUq5",

      "media_url" : "http://pbs.twimg.com/media/D9V6hdTXsAAgTGr.jpg",

      "id_str" : "1140952795828432896",

      "id" : "1140952795828432896",

      "media_url_https" : "https://pbs.twimg.com/media/D9V6hdTXsAAgTGr.jpg",

      "sizes" : {

        "small" : {

          "w" : "680",

          "h" : "680",

          "resize" : "fit"

        },

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "large" : {

          "w" : "960",

          "h" : "960",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "960",

          "h" : "960",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "display_url" : "pic.twitter.com/GToDd8hUq5"

    } ],

    "hashtags" : [ ]

  },

  "display_text_range" : [ "0", "100" ],

  "favorite_count" : "0",

  "id_str" : "1140952800697995266",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140952800697995266",

  "possibly_sensitive" : false,

  "created_at" : "Tue Jun 18 12:02:00 +0000 2019",

  "favorited" : false,

  "full_text" : "KONSERVATIVA TÄNKER MER RATIONELLT OCH STRUNTAR I KÄNSLOR OCH SÅNT FJANT...! https://t.co/GToDd8hUq5",

  "lang" : "sv",

  "extended_entities" : {

    "media" : [ {

      "expanded_url" : "https://twitter.com/davidraxen/status/1140952800697995266/photo/1",

      "indices" : [ "77", "100" ],

      "url" : "https://t.co/GToDd8hUq5",

      "media_url" : "http://pbs.twimg.com/media/D9V6hdTXsAAgTGr.jpg",

      "id_str" : "1140952795828432896",

      "id" : "1140952795828432896",

      "media_url_https" : "https://pbs.twimg.com/media/D9V6hdTXsAAgTGr.jpg",

      "sizes" : {

        "small" : {

          "w" : "680",

          "h" : "680",

          "resize" : "fit"

        },

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "large" : {

          "w" : "960",

          "h" : "960",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "960",

          "h" : "960",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "display_url" : "pic.twitter.com/GToDd8hUq5"

    } ]

  }

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Joakim Zander",

      "screen_name" : "JoakimZander",

      "indices" : [ "3", "16" ],

      "id_str" : "521650971",

      "id" : "521650971"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "140" ],

  "favorite_count" : "0",

  "id_str" : "1140941647720718336",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140941647720718336",

  "created_at" : "Tue Jun 18 11:17:41 +0000 2019",

  "favorited" : false,

  "full_text" : "RT @JoakimZander: Det borde väl stå rätt klart nu att folks ”oro över samhällsutvecklingen” inte handlar om kriminalitet, ekonomi eller isl…",

  "lang" : "sv"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Filippa Mannerheim",

      "screen_name" : "FMannerheim",

      "indices" : [ "0", "12" ],

      "id_str" : "1256602400",

      "id" : "1256602400"

    }, {

      "name" : "Alice Teodorescu Måwe",

      "screen_name" : "alicemedce",

      "indices" : [ "13", "24" ],

      "id_str" : "273134386",

      "id" : "273134386"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "209" ],

  "favorite_count" : "0",

  "in_reply_to_status_id_str" : "1140109736391663616",

  "id_str" : "1140694443739222017",

  "in_reply_to_user_id" : "1256602400",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140694443739222017",

  "in_reply_to_status_id" : "1140109736391663616",

  "created_at" : "Mon Jun 17 18:55:23 +0000 2019",

  "favorited" : false,

  "full_text" : "@FMannerheim @alicemedce Inte då. En annan har bara bott i knegarort (Smedjebacken), Ryd i Linköping (och då inte i studentdelen) och Fruängen (som Jimmie minsann hävdat står i lågor) och jag raljerade friskt.",

  "lang" : "sv",

  "in_reply_to_screen_name" : "FMannerheim",

  "in_reply_to_user_id_str" : "1256602400"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Patrik Brenning",

      "screen_name" : "PatrikBrenning",

      "indices" : [ "0", "15" ],

      "id_str" : "291709255",

      "id" : "291709255"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "89" ],

  "favorite_count" : "0",

  "in_reply_to_status_id_str" : "1140543081609158656",

  "id_str" : "1140549219188649985",

  "in_reply_to_user_id" : "291709255",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140549219188649985",

  "in_reply_to_status_id" : "1140543081609158656",

  "created_at" : "Mon Jun 17 09:18:18 +0000 2019",

  "favorited" : false,

  "full_text" : "@PatrikBrenning Knepigt när Barcelona och Real är enda destinationen för superspelare...!",

  "lang" : "sv",

  "in_reply_to_screen_name" : "PatrikBrenning",

  "in_reply_to_user_id_str" : "291709255"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Teodor Koistinen",

      "screen_name" : "TeodorKoistinen",

      "indices" : [ "0", "16" ],

      "id_str" : "2811008905",

      "id" : "2811008905"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "260" ],

  "favorite_count" : "2",

  "in_reply_to_status_id_str" : "1140528840579649536",

  "id_str" : "1140530929380802560",

  "in_reply_to_user_id" : "118417712",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140530929380802560",

  "in_reply_to_status_id" : "1140528840579649536",

  "created_at" : "Mon Jun 17 08:05:38 +0000 2019",

  "favorited" : false,

  "full_text" : "@TeodorKoistinen Eller verkligheten att vi går in i en helt ny era med automation och ai som kan (kommer) förändra synen på arbete helt (när två nya händer inte längre betyder ökad produktion) och det är quasi-fascister som styr alla världens stora länder? O.o",

  "lang" : "sv",

  "in_reply_to_screen_name" : "davidraxen",

  "in_reply_to_user_id_str" : "118417712"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Teodor Koistinen",

      "screen_name" : "TeodorKoistinen",

      "indices" : [ "0", "16" ],

      "id_str" : "2811008905",

      "id" : "2811008905"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "173" ],

  "favorite_count" : "3",

  "in_reply_to_status_id_str" : "1140488388891611136",

  "id_str" : "1140528840579649536",

  "in_reply_to_user_id" : "2811008905",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140528840579649536",

  "in_reply_to_status_id" : "1140488388891611136",

  "created_at" : "Mon Jun 17 07:57:20 +0000 2019",

  "favorited" : false,

  "full_text" : "@TeodorKoistinen Verkligheten att mänskligheten verkar stå inför den största folkvandringen i sin historia i och med klimatet och demografiförändringarna i Asien och Afrika?",

  "lang" : "sv",

  "in_reply_to_screen_name" : "TeodorKoistinen",

  "in_reply_to_user_id_str" : "2811008905"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Peter Wolodarski",

      "screen_name" : "pwolodarski",

      "indices" : [ "0", "12" ],

      "id_str" : "188282409",

      "id" : "188282409"

    }, {

      "name" : "Ola Spännar",

      "screen_name" : "olaspannar",

      "indices" : [ "13", "24" ],

      "id_str" : "19305444",

      "id" : "19305444"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "44" ],

  "favorite_count" : "0",

  "in_reply_to_status_id_str" : "1140354772819750912",

  "id_str" : "1140356463157813248",

  "in_reply_to_user_id" : "188282409",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140356463157813248",

  "in_reply_to_status_id" : "1140354772819750912",

  "created_at" : "Sun Jun 16 20:32:22 +0000 2019",

  "favorited" : false,

  "full_text" : "@pwolodarski @olaspannar *invandringskritik.",

  "lang" : "no",

  "in_reply_to_screen_name" : "pwolodarski",

  "in_reply_to_user_id_str" : "188282409"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "مارتنDr Martin Estvall",

      "screen_name" : "DrEstvall",

      "indices" : [ "3", "13" ],

      "id_str" : "707540316",

      "id" : "707540316"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "140" ],

  "favorite_count" : "0",

  "id_str" : "1140355915574632449",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140355915574632449",

  "created_at" : "Sun Jun 16 20:30:11 +0000 2019",

  "favorited" : false,

  "full_text" : "RT @DrEstvall: Många rädda tanter och farbröder är yngre än mig, men det uppenbara måste tydligen sägas: huvudorsaken till att ni känner er…",

  "lang" : "sv"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Benjamin Dixon",

      "screen_name" : "BenjaminPDixon",

      "indices" : [ "3", "18" ],

      "id_str" : "2949261587",

      "id" : "2949261587"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "139" ],

  "favorite_count" : "0",

  "id_str" : "1140232752278716417",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140232752278716417",

  "created_at" : "Sun Jun 16 12:20:47 +0000 2019",

  "favorited" : false,

  "full_text" : "RT @BenjaminPDixon: People try so hard to explain racism  bigotry with any theory except actual hatred. \n\n“Rich racists are only racist to…",

  "lang" : "en"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Lisa Förare Winbladh",

      "screen_name" : "foraretaffel",

      "indices" : [ "3", "16" ],

      "id_str" : "18770497",

      "id" : "18770497"

    }, {

      "name" : "Agnes Wold",

      "screen_name" : "AgnesWold",

      "indices" : [ "18", "28" ],

      "id_str" : "1400777804",

      "id" : "1400777804"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "139" ],

  "favorite_count" : "0",

  "id_str" : "1140223403951083520",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140223403951083520",

  "created_at" : "Sun Jun 16 11:43:38 +0000 2019",

  "favorited" : false,

  "full_text" : "RT @foraretaffel: @AgnesWold Ja, det finns en så obehaglig naiv syn på att rasistiska idéer är intellektuellt spännande diskussionsämnen –…",

  "lang" : "sv"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Kajsa Dovstad",

      "screen_name" : "kajsadovstad",

      "indices" : [ "0", "13" ],

      "id_str" : "558486260",

      "id" : "558486260"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "60" ],

  "favorite_count" : "6",

  "in_reply_to_status_id_str" : "1140157314927878144",

  "id_str" : "1140161326246977537",

  "in_reply_to_user_id" : "558486260",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140161326246977537",

  "in_reply_to_status_id" : "1140157314927878144",

  "created_at" : "Sun Jun 16 07:36:58 +0000 2019",

  "favorited" : false,

  "full_text" : "@kajsadovstad Ja det var onekligen vad kritiken handlade om.",

  "lang" : "sv",

  "in_reply_to_screen_name" : "kajsadovstad",

  "in_reply_to_user_id_str" : "558486260"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Jörgen Huitfeldt",

      "screen_name" : "JHuitfeldt",

      "indices" : [ "0", "11" ],

      "id_str" : "392683955",

      "id" : "392683955"

    }, {

      "name" : "Kajsa Dovstad",

      "screen_name" : "kajsadovstad",

      "indices" : [ "12", "25" ],

      "id_str" : "558486260",

      "id" : "558486260"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "106" ],

  "favorite_count" : "2",

  "in_reply_to_status_id_str" : "1139882974139359233",

  "id_str" : "1140144662314913792",

  "in_reply_to_user_id" : "392683955",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140144662314913792",

  "in_reply_to_status_id" : "1139882974139359233",

  "created_at" : "Sun Jun 16 06:30:45 +0000 2019",

  "favorited" : false,

  "full_text" : "@JHuitfeldt @kajsadovstad Sjukt att man inte ens får skriva rasistiskt dravel utan att nån protesterar än.",

  "lang" : "sv",

  "in_reply_to_screen_name" : "JHuitfeldt",

  "in_reply_to_user_id_str" : "392683955"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "user_mentions" : [ {

      "name" : "Catharina",

      "screen_name" : "Newbergskan",

      "indices" : [ "3", "15" ],

      "id_str" : "2607003269",

      "id" : "2607003269"

    } ],

    "urls" : [ ],

    "symbols" : [ ],

    "media" : [ {

      "expanded_url" : "https://twitter.com/Newbergskan/status/1139892460023406594/photo/1",

      "source_status_id" : "1139892460023406594",

      "indices" : [ "76", "99" ],

      "url" : "https://t.co/RVpj7uEbTx",

      "media_url" : "http://pbs.twimg.com/media/D9G2I30WsAA3fbN.jpg",

      "id_str" : "1139892444240195584",

      "source_user_id" : "2607003269",

      "id" : "1139892444240195584",

      "media_url_https" : "https://pbs.twimg.com/media/D9G2I30WsAA3fbN.jpg",

      "source_user_id_str" : "2607003269",

      "sizes" : {

        "small" : {

          "w" : "635",

          "h" : "680",

          "resize" : "fit"

        },

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "large" : {

          "w" : "1063",

          "h" : "1139",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "1063",

          "h" : "1139",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "source_status_id_str" : "1139892460023406594",

      "display_url" : "pic.twitter.com/RVpj7uEbTx"

    } ],

    "hashtags" : [ ]

  },

  "display_text_range" : [ "0", "99" ],

  "favorite_count" : "0",

  "id_str" : "1140144177323433984",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1140144177323433984",

  "possibly_sensitive" : false,

  "created_at" : "Sun Jun 16 06:28:49 +0000 2019",

  "favorited" : false,

  "full_text" : "RT @Newbergskan: Vet att ni alla vet det här redan men lägger den här ändå: https://t.co/RVpj7uEbTx",

  "lang" : "sv",

  "extended_entities" : {

    "media" : [ {

      "expanded_url" : "https://twitter.com/Newbergskan/status/1139892460023406594/photo/1",

      "source_status_id" : "1139892460023406594",

      "indices" : [ "76", "99" ],

      "url" : "https://t.co/RVpj7uEbTx",

      "media_url" : "http://pbs.twimg.com/media/D9G2I30WsAA3fbN.jpg",

      "id_str" : "1139892444240195584",

      "source_user_id" : "2607003269",

      "id" : "1139892444240195584",

      "media_url_https" : "https://pbs.twimg.com/media/D9G2I30WsAA3fbN.jpg",

      "source_user_id_str" : "2607003269",

      "sizes" : {

        "small" : {

          "w" : "635",

          "h" : "680",

          "resize" : "fit"

        },

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "large" : {

          "w" : "1063",

          "h" : "1139",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "1063",

          "h" : "1139",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "source_status_id_str" : "1139892460023406594",

      "display_url" : "pic.twitter.com/RVpj7uEbTx"

    }, {

      "expanded_url" : "https://twitter.com/Newbergskan/status/1139892460023406594/photo/1",

      "source_status_id" : "1139892460023406594",

      "indices" : [ "76", "99" ],

      "url" : "https://t.co/RVpj7uEbTx",

      "media_url" : "http://pbs.twimg.com/media/D9G2JWhXkAEPb1V.jpg",

      "id_str" : "1139892452482060289",

      "source_user_id" : "2607003269",

      "id" : "1139892452482060289",

      "media_url_https" : "https://pbs.twimg.com/media/D9G2JWhXkAEPb1V.jpg",

      "source_user_id_str" : "2607003269",

      "sizes" : {

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "small" : {

          "w" : "680",

          "h" : "577",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "1028",

          "h" : "873",

          "resize" : "fit"

        },

        "large" : {

          "w" : "1028",

          "h" : "873",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "source_status_id_str" : "1139892460023406594",

      "display_url" : "pic.twitter.com/RVpj7uEbTx"

    } ]

  }

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "201" ],

  "favorite_count" : "0",

  "id_str" : "1139952432677621760",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1139952432677621760",

  "created_at" : "Sat Jun 15 17:46:53 +0000 2019",

  "favorited" : false,

  "full_text" : "Hur många ledarkrönikor med lätt rasistisk underton i GP, SvD och DI (och då och då Expressen..!) krävs för att vi åtminstone kan begrava myterna gällande åsiktskorridåren och medias ”vänstervridning”?",

  "lang" : "sv"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Patrik Syk",

      "screen_name" : "PatrikSyk",

      "indices" : [ "0", "10" ],

      "id_str" : "23177985",

      "id" : "23177985"

    }, {

      "name" : "Erik Niva",

      "screen_name" : "ErikNiva",

      "indices" : [ "49", "58" ],

      "id_str" : "154820247",

      "id" : "154820247"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "197" ],

  "favorite_count" : "1",

  "in_reply_to_status_id_str" : "1139546352378138626",

  "id_str" : "1139566216824184835",

  "in_reply_to_user_id" : "23177985",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1139566216824184835",

  "in_reply_to_status_id" : "1139546352378138626",

  "created_at" : "Fri Jun 14 16:12:12 +0000 2019",

  "favorited" : false,

  "full_text" : "@PatrikSyk Väntar fortfarande på min medalj från @ErikNiva sen förra gången! - Även om jag nog måste backa lite på mitt insinuerande om att Foppas karriär skulle bli kalas bara han stannade i RB...",

  "lang" : "sv",

  "in_reply_to_screen_name" : "PatrikSyk",

  "in_reply_to_user_id_str" : "23177985"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Nils Karlsson",

      "screen_name" : "FilosofenNils",

      "indices" : [ "0", "14" ],

      "id_str" : "467291739",

      "id" : "467291739"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "190" ],

  "favorite_count" : "2",

  "in_reply_to_status_id_str" : "1139247244534853632",

  "id_str" : "1139249391011860480",

  "in_reply_to_user_id" : "467291739",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1139249391011860480",

  "in_reply_to_status_id" : "1139247244534853632",

  "created_at" : "Thu Jun 13 19:13:15 +0000 2019",

  "favorited" : false,

  "full_text" : "@FilosofenNils Fast ska man vara på den nivån så är vi ett flockdjur som klarar av ~150 personliga relationer. Alla andra (som nationen t.ex.) är ju bara något vi gemensamt låtsas existerar.",

  "lang" : "sv",

  "in_reply_to_screen_name" : "FilosofenNils",

  "in_reply_to_user_id_str" : "467291739"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "user_mentions" : [ {

      "name" : "Vulgaris Borealis \uD83E\uDD61",

      "screen_name" : "Unruh20",

      "indices" : [ "3", "11" ],

      "id_str" : "992739359253303301",

      "id" : "992739359253303301"

    }, {

      "name" : "Peter Arnott",

      "screen_name" : "PeterArnottGlas",

      "indices" : [ "13", "29" ],

      "id_str" : "2396520638",

      "id" : "2396520638"

    }, {

      "name" : "Billy Bragg",

      "screen_name" : "billybragg",

      "indices" : [ "30", "41" ],

      "id_str" : "13496142",

      "id" : "13496142"

    } ],

    "urls" : [ ],

    "symbols" : [ ],

    "media" : [ {

      "expanded_url" : "https://twitter.com/Unruh20/status/1138465316693000194/photo/1",

      "source_status_id" : "1138465316693000194",

      "indices" : [ "42", "65" ],

      "url" : "https://t.co/YBHE0Kkr2n",

      "media_url" : "http://pbs.twimg.com/media/D8ykKgxX4AAHOyh.jpg",

      "id_str" : "1138465306320560128",

      "source_user_id" : "992739359253303301",

      "id" : "1138465306320560128",

      "media_url_https" : "https://pbs.twimg.com/media/D8ykKgxX4AAHOyh.jpg",

      "source_user_id_str" : "992739359253303301",

      "sizes" : {

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "small" : {

          "w" : "485",

          "h" : "680",

          "resize" : "fit"

        },

        "large" : {

          "w" : "1142",

          "h" : "1600",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "857",

          "h" : "1200",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "source_status_id_str" : "1138465316693000194",

      "display_url" : "pic.twitter.com/YBHE0Kkr2n"

    } ],

    "hashtags" : [ ]

  },

  "display_text_range" : [ "0", "65" ],

  "favorite_count" : "0",

  "id_str" : "1138675677534797825",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1138675677534797825",

  "possibly_sensitive" : false,

  "created_at" : "Wed Jun 12 05:13:31 +0000 2019",

  "favorited" : false,

  "full_text" : "RT @Unruh20: @PeterArnottGlas @billybragg https://t.co/YBHE0Kkr2n",

  "lang" : "und",

  "extended_entities" : {

    "media" : [ {

      "expanded_url" : "https://twitter.com/Unruh20/status/1138465316693000194/photo/1",

      "source_status_id" : "1138465316693000194",

      "indices" : [ "42", "65" ],

      "url" : "https://t.co/YBHE0Kkr2n",

      "media_url" : "http://pbs.twimg.com/media/D8ykKgxX4AAHOyh.jpg",

      "id_str" : "1138465306320560128",

      "source_user_id" : "992739359253303301",

      "id" : "1138465306320560128",

      "media_url_https" : "https://pbs.twimg.com/media/D8ykKgxX4AAHOyh.jpg",

      "source_user_id_str" : "992739359253303301",

      "sizes" : {

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "small" : {

          "w" : "485",

          "h" : "680",

          "resize" : "fit"

        },

        "large" : {

          "w" : "1142",

          "h" : "1600",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "857",

          "h" : "1200",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "source_status_id_str" : "1138465316693000194",

      "display_url" : "pic.twitter.com/YBHE0Kkr2n"

    } ]

  }

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ {

      "name" : "Per Holfve",

      "screen_name" : "Juristfan",

      "indices" : [ "3", "13" ],

      "id_str" : "749257939",

      "id" : "749257939"

    } ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "139" ],

  "favorite_count" : "0",

  "id_str" : "1137109789073444864",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1137109789073444864",

  "created_at" : "Fri Jun 07 21:31:14 +0000 2019",

  "favorited" : false,

  "full_text" : "RT @Juristfan: Nyamko Sabuni får mig att tänka på detta gyllene citat av Groucho Marx:\n\n\"Jag håller hårt på mina principer och om dom inte…",

  "lang" : "sv"

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "user_mentions" : [ {

      "name" : "Patrik Syk",

      "screen_name" : "PatrikSyk",

      "indices" : [ "0", "10" ],

      "id_str" : "23177985",

      "id" : "23177985"

    } ],

    "urls" : [ ],

    "symbols" : [ ],

    "media" : [ {

      "expanded_url" : "https://twitter.com/davidraxen/status/1137100225032704002/photo/1",

      "indices" : [ "81", "104" ],

      "url" : "https://t.co/2WCPV5upy1",

      "media_url" : "http://pbs.twimg.com/media/D8fKn5kWwAAXB3R.jpg",

      "id_str" : "1137100217751355392",

      "id" : "1137100217751355392",

      "media_url_https" : "https://pbs.twimg.com/media/D8fKn5kWwAAXB3R.jpg",

      "sizes" : {

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "small" : {

          "w" : "680",

          "h" : "680",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "1200",

          "h" : "1200",

          "resize" : "fit"

        },

        "large" : {

          "w" : "1936",

          "h" : "1936",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "display_url" : "pic.twitter.com/2WCPV5upy1"

    } ],

    "hashtags" : [ ]

  },

  "display_text_range" : [ "0", "104" ],

  "favorite_count" : "1",

  "in_reply_to_status_id_str" : "1137065963248111616",

  "id_str" : "1137100225032704002",

  "in_reply_to_user_id" : "23177985",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1137100225032704002",

  "in_reply_to_status_id" : "1137065963248111616",

  "possibly_sensitive" : false,

  "created_at" : "Fri Jun 07 20:53:14 +0000 2019",

  "favorited" : false,

  "full_text" : "@PatrikSyk Kan bli Jävligt balanserat om man bara ger upp allt annat i livet...! https://t.co/2WCPV5upy1",

  "lang" : "sv",

  "in_reply_to_screen_name" : "PatrikSyk",

  "in_reply_to_user_id_str" : "23177985",

  "extended_entities" : {

    "media" : [ {

      "expanded_url" : "https://twitter.com/davidraxen/status/1137100225032704002/photo/1",

      "indices" : [ "81", "104" ],

      "url" : "https://t.co/2WCPV5upy1",

      "media_url" : "http://pbs.twimg.com/media/D8fKn5kWwAAXB3R.jpg",

      "id_str" : "1137100217751355392",

      "id" : "1137100217751355392",

      "media_url_https" : "https://pbs.twimg.com/media/D8fKn5kWwAAXB3R.jpg",

      "sizes" : {

        "thumb" : {

          "w" : "150",

          "h" : "150",

          "resize" : "crop"

        },

        "small" : {

          "w" : "680",

          "h" : "680",

          "resize" : "fit"

        },

        "medium" : {

          "w" : "1200",

          "h" : "1200",

          "resize" : "fit"

        },

        "large" : {

          "w" : "1936",

          "h" : "1936",

          "resize" : "fit"

        }

      },

      "type" : "photo",

      "display_url" : "pic.twitter.com/2WCPV5upy1"

    } ]

  }

}, {

  "retweeted" : false,

  "source" : "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",

  "entities" : {

    "hashtags" : [ ],

    "symbols" : [ ],

    "user_mentions" : [ ],

    "urls" : [ ]

  },

  "display_text_range" : [ "0", "92" ],

  "favorite_count" : "0",

  "id_str" : "1136553136074764289",

  "truncated" : false,

  "retweet_count" : "0",

  "id" : "1136553136074764289",

  "created_at" : "Thu Jun 06 08:39:18 +0000 2019",

  "favorited" : false,

  "full_text" : "Om någon undrade så anser jag fortfarande att nationaldagen bör flyttas till midsommarafton.",

  "lang" : "sv"

} ]

'''
twitter_data = json.loads(tweets)



#print(twitter_data[0].keys())



#keys to delete

keys = ['retweeted', 'source', 'entities', 'display_text_range', 'id_str','truncated','id', 'possibly_sensitive','favorited','extended_entities', 'in_reply_to_status_id', 'in_reply_to_status_id_str','in_reply_to_user_id', 'in_reply_to_user_id_str']

df = pd.DataFrame()

for dictionary in twitter_data:

    for key in keys:

        try:

            del dictionary[key]

        except KeyError:

           pass

    df = df.append(pd.DataFrame.from_dict([dictionary]), sort=False)

del keys

del dictionary

del key

    

#- Split dates.    

A = df["created_at"].str.split(" ", expand = True)

A = A.drop(4, axis=1)

AHeaders = ["Year", "Month", "Day"]

A.columns = ["Weekday", "Month", "Day", "Time", "Year"]

df = pd.concat([df, A], axis=1)

df = df.drop("created_at", axis=1)

del A

del AHeaders



#- Remove mentions from tweets.

for words in df["full_text"]:

    ssplit2 = ""

    ssplit = words.split()

    for word in ssplit:

        if word[0] == "@":

            word = ""

        else:

            ssplit2 = ssplit2 + " " + word

    if ssplit2[0] == " ":

        ssplit2 = ssplit2[1:]    

    else:

        ssplit2 = ssplit2                

    df.loc[df.full_text == words, "full_text"] = ssplit2  

del words

del ssplit2

del ssplit

del word

#----- Make integers out of strings in numerical columns

for word in df["favorite_count"]:

    df.loc[df.favorite_count == word, "favorite_count"] = int(word)



df["Hour"] = df["Time"].apply(lambda hour: hour[:2]+":00")       
df = pd.read_csv('../input/twitter/Twitter_Data.csv', sep = ",")
plt.figure(figsize=(10,10))

df.Weekday.value_counts().plot.pie(autopct='%1.1f%%',shadow=True,cmap='brg')
plt.figure(figsize=(15,10))

g = sns.countplot(x="Hour", data=df, order = ["00:00", "01:00", "02:00", "03:00", "04:00","05:00", "06:00", "07:00", "08:00", "09:00",

                                                        "10:00", "11:00", "12:00", "13:00", "14:00",

                                                        "15:00", "16:00", "17:00", "18:00", "19:00",

                                                        "20:00", "21:00", "22:00", "23:00"])

g.set_xticklabels(g.get_xticklabels(),rotation=30)
print(df.loc[df.Hour == "04:00", "full_text"])

print(df.loc[df.Hour == "04:00", "Year"])
Years = df["Year"].unique().tolist()

df_temp = df.groupby(["Month"]).apply(lambda column: column["Year"].value_counts()).unstack().reindex(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]).reset_index()





df_temp.plot(x="Month", y=Years, kind="area", figsize=(20, 10))

plt.figure(figsize=(10,10))

sns.countplot(x="Year", data=df)
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



sns.countplot(x="favorite_count", data=df, order=df['favorite_count'].value_counts().index, ax=ax1)

plt.xlabel("Favorites")

sns.countplot(x="lang", data=df, order=df['lang'].value_counts().iloc[:5].index, ax=ax2)

plt.xlabel("Language")

sns.countplot(x="retweet_count", data=df, order=df['retweet_count'].value_counts().index, ax=ax3)

plt.xlabel("Retweets")

sns.countplot(x="in_reply_to_screen_name", data=df, order=df["in_reply_to_screen_name"].value_counts().iloc[:10].index, ax=ax4)

plt.xticks(rotation=30)
RTs = 0

MyTs = 0

#- Find RTs.

for word in df["full_text"]:

    if word[:2] == "RT":

        RTs = RTs + 1

    else:

        MyTs = MyTs + 1 

df_temp = {'Retweets': [RTs], 

        'My_tweets': [MyTs]}

df_temp = pd.DataFrame(df_temp, columns = ['Retweets', 'My_tweets']).reset_index()

del RTs

del MyTs     



df_temp.plot(x="index", y=["Retweets", "My_tweets"], kind="bar")
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=120, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'som', 'att', 'det', 'är', 'RT', 'https'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(text)

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');
comments_text = str(df.full_text)

comments_mask = np.array(Image.open('../input/davidr/IMG_1065.png'))

plot_wordcloud(comments_text, comments_mask, max_words=400, max_font_size=120, 

               title = 'Most frequent words in tweets', title_size=25)