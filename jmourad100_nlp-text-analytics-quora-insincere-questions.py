import os, math, re, time
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import ( Dense, Input, Conv1D, Conv2D, MaxPool2D, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D, 
                           Activation, Dropout, SpatialDropout1D, Embedding, Concatenate, concatenate, Reshape, Flatten, 
                           CuDNNLSTM, CuDNNGRU, Bidirectional )
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

import tensorflow as tf

t_start = time.time()
embed_size = 300      # how big is each word vector
max_features = 120000 #95000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70           #70 # max number of words in a question to use
nb_features = 6
EPOCHS = 10
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '~', '•',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not", "can't" : "cannot", "couldn't" : "could not", "didn't" : "did not", "doesn't" : "does not",
"don't" : "do not", "hadn't" : "had not", "hasn't" : "has not", "haven't" : "have not", "he'd" : "he would", "he'll" : "he will",
"he's" : "he is", "i'd" : "I would", "i'd" : "I had", "i'll" : "I will", "i'm" : "I am", "isn't" : "is not", "it's" : "it is",
"it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not", "mustn't" : "must not", "shan't" : "shall not",
"she'd" : "she would", "she'll" : "she will", "she's" : "she is", "shouldn't" : "should not", "that's" : "that is", "there's" : "there is",
"they'd" : "they would", "they'll" : "they will", "they're" : "they are", "they've" : "they have", "we'd" : "we would", "we're" : "we are",
"weren't" : "were not", "we've" : "we have", "what'll" : "what will", "what're" : "what are", "what's" : "what is", "what've" : "what have",
"where's" : "where is", "who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is", "who've" : "who have",
"won't" : "will not", "wouldn't" : "would not", "you'd" : "you would", "you'll" : "you will", "you're" : "you are", "you've" : "you have",
"'re": " are", "wasn't": "was not", "we'll":" will", "didn't": "did not", "tryin'":"trying"}

# bad_words = "2 girls 1 cup, 2g1c, 4r5e, 5h1t, 5hit, a$$, a$$hole, a_s_s, a2m, a54, a55, a55hole, acrotomophilia, aeolus, ahole, alabama hot pocket, alaskan pipeline, anal, anal impaler, anal leakage, analprobe, anilingus, anus, apeshit, ar5e, areola, areole, arian, arrse, arse, arsehole, aryan, ass, ass fuck, ass fuck, ass hole, assbag, assbandit, assbang, assbanged, assbanger, assbangs, assbite, assclown, asscock, asscracker, asses, assface, assfaces, assfuck, assfucker, ass-fucker, assfukka, assgoblin, assh0le, asshat, ass-hat, asshead, assho1e, asshole, assholes, asshopper, ass-jabber, assjacker, asslick, asslicker, assmaster, assmonkey, assmucus, assmucus, assmunch, assmuncher, assnigger, asspirate, ass-pirate, assshit, assshole, asssucker, asswad, asswhole, asswipe, asswipes, auto erotic, autoerotic, axwound, azazel, azz, b!tch, b00bs, b17ch, b1tch, babeland, baby batter, baby juice, ball gag, ball gravy, ball kicking, ball licking, ball sack, ball sucking, ballbag, balls, ballsack, bampot, bang (one's) box, bangbros, bareback, barely legal, barenaked, barf, bastard, bastardo, bastards, bastinado, batty boy, bawdy, bbw, bdsm, beaner, beaners, beardedclam, beastial, beastiality, beatch, beaver, beaver cleaver, beaver lips, beef curtain, beef curtain, beef curtains, beeyotch, bellend, bender, beotch, bescumber, bestial, bestiality, bi+ch, biatch, big black, big breasts, big knockers, big tits, bigtits, bimbo, bimbos, bint, birdlock, bitch, bitch tit, bitch tit, bitchass, bitched, bitcher, bitchers, bitches, bitchin, bitching, bitchtits, bitchy, black cock, blonde action, blonde on blonde action, bloodclaat, bloody, bloody hell, blow job, blow me, blow mud, blow your load, blowjob, blowjobs, blue waffle, blue waffle, blumpkin, blumpkin, bod, bodily, boink, boiolas, bollock, bollocks, bollok, bollox, bondage, boned, boner, boners, bong, boob, boobies, boobs, booby, booger, bookie, boong, booobs, boooobs, booooobs, booooooobs, bootee, bootie, booty, booty call, booze, boozer, boozy, bosom, bosomy, breasts, Breeder, brotherfucker, brown showers, brunette action, buceta, bugger, bukkake, bull shit, bulldyke, bullet vibe, bullshit, bullshits, bullshitted, bullturds, bum, bum boy, bumblefuck, bumclat, bummer, buncombe, bung, bung hole, bunghole, bunny fucker, bust a load, bust a load, busty, butt, butt fuck, butt fuck, butt plug, buttcheeks, buttfuck, buttfucka, buttfucker, butthole, buttmuch, buttmunch, butt-pirate, buttplug, c.0.c.k, c.o.c.k., c.u.n.t, c0ck, c-0-c-k, c0cksucker, caca, cacafuego, cahone, camel toe, cameltoe, camgirl, camslut, camwhore, carpet muncher, carpetmuncher, cawk, cervix, chesticle, chi-chi man, chick with a dick, child-fucker, chinc, chincs, chink, chinky, choad, choade, choade, choc ice, chocolate rosebuds, chode, chodes, chota bags, chota bags, cipa, circlejerk, cl1t, cleveland steamer, climax, clit, clit licker, clit licker, clitface, clitfuck, clitoris, clitorus, clits, clitty, clitty litter, clitty litter, clover clamps, clunge, clusterfuck, cnut, cocain, cocaine, coccydynia, cock, c-o-c-k, cock pocket, cock pocket, cock snot, cock snot, cock sucker, cockass, cockbite, cockblock, cockburger, cockeye, cockface, cockfucker, cockhead, cockholster, cockjockey, cockknocker, cockknoker, Cocklump, cockmaster, cockmongler, cockmongruel, cockmonkey, cockmunch, cockmuncher, cocknose, cocknugget, cocks, cockshit, cocksmith, cocksmoke, cocksmoker, cocksniffer, cocksuck, cocksuck, cocksucked, cocksucked, cocksucker, cock-sucker, cocksuckers, cocksucking, cocksucks, cocksucks, cocksuka, cocksukka, cockwaffle, coffin dodger, coital, cok, cokmuncher, coksucka, commie, condom, coochie, coochy, coon, coonnass, coons, cooter, cop some wood, cop some wood, coprolagnia, coprophilia, corksucker, cornhole, cornhole, corp whore, corp whore, corpulent, cox, crabs, crack, cracker, crackwhore, crap, crappy, creampie, cretin, crikey, cripple, crotte, cum, cum chugger, cum chugger, cum dumpster, cum dumpster, cum freak, cum freak, cum guzzler, cum guzzler, cumbubble, cumdump, cumdump, cumdumpster, cumguzzler, cumjockey, cummer, cummin, cumming, cums, cumshot, cumshots, cumslut, cumstain, cumtart, cunilingus, cunillingus, cunnie, cunnilingus, cunny, cunt, c-u-n-t, cunt hair, cunt hair, cuntass, cuntbag, cuntbag, cuntface, cunthole, cunthunter, cuntlick, cuntlick, cuntlicker, cuntlicker, cuntlicking, cuntlicking, cuntrag, cunts, cuntsicle, cuntsicle, cuntslut, cunt-struck, cunt-struck, cus, cut rope, cut rope, cyalis, cyberfuc, cyberfuck, cyberfuck, cyberfucked, cyberfucked, cyberfucker, cyberfuckers, cyberfucking, cyberfucking, d0ng, d0uch3, d0uche, d1ck, d1ld0, d1ldo, dago, dagos, dammit, damn, damned, damnit, darkie, darn, date rape, daterape, dawgie-style, deep throat, deepthroat, deggo, dendrophilia, dick, dick head, dick hole, dick hole, dick shy, dick shy, dickbag, dickbeaters, dickdipper, dickface, dickflipper, dickfuck, dickfucker, dickhead, dickheads, dickhole, dickish, dick-ish, dickjuice, dickmilk, dickmonger, dickripper, dicks, dicksipper, dickslap, dick-sneeze, dicksucker, dicksucking, dicktickler, dickwad, dickweasel, dickweed, dickwhipper, dickwod, dickzipper, diddle, dike, dildo, dildos, diligaf, dillweed, dimwit, dingle, dingleberries, dingleberry, dink, dinks, dipship, dipshit, dirsa, dirty, dirty pillows, dirty sanchez, dirty Sanchez, div, dlck, dog style, dog-fucker, doggie style, doggiestyle, doggie-style, doggin, dogging, doggy style, doggystyle, doggy-style, dolcett, domination, dominatrix, dommes, dong, donkey punch, donkeypunch, donkeyribber, doochbag, doofus, dookie, doosh, dopey, double dong, double penetration, Doublelift, douch3, douche, douchebag, douchebags, douche-fag, douchewaffle, douchey, dp action, drunk, dry hump, duche, dumass, dumb ass, dumbass, dumbasses, Dumbcunt, dumbfuck, dumbshit, dummy, dumshit, dvda, dyke, dykes, eat a dick, eat a dick, eat hair pie, eat hair pie, eat my ass, ecchi, ejaculate, ejaculated, ejaculates, ejaculates, ejaculating, ejaculating, ejaculatings, ejaculation, ejakulate, erect, erection, erotic, erotism, escort, essohbee, eunuch, extacy, extasy, f u c k, f u c k e r, f.u.c.k, f_u_c_k, f4nny, facial, fack, fag, fagbag, fagfucker, fagg, fagged, fagging, faggit, faggitt, faggot, faggotcock, faggots, faggs, fagot, fagots, fags, fagtard, faig, faigt, fanny, fannybandit, fannyflaps, fannyfucker, fanyy, fart, fartknocker, fatass, fcuk, fcuker, fcuking, fecal, feck, fecker, feist, felch, felcher, felching, fellate, fellatio, feltch, feltcher, female squirting, femdom, fenian, fice, figging, fingerbang, fingerfuck, fingerfuck, fingerfucked, fingerfucked, fingerfucker, fingerfucker, fingerfuckers, fingerfucking, fingerfucking, fingerfucks, fingerfucks, fingering, fist fuck, fist fuck, fisted, fistfuck, fistfucked, fistfucked, fistfucker, fistfucker, fistfuckers, fistfuckers, fistfucking, fistfucking, fistfuckings, fistfuckings, fistfucks, fistfucks, fisting, fisty, flamer, flange, flaps, fleshflute, flog the log, flog the log, floozy, foad, foah, fondle, foobar, fook, fooker, foot fetish, footjob, foreskin, freex, frenchify, frigg, frigga, frotting, fubar, fuc, fuck, fuck, f-u-c-k, fuck buttons, fuck hole, fuck hole, Fuck off, fuck puppet, fuck puppet, fuck trophy, fuck trophy, fuck yo mama, fuck yo mama, fuck you, fucka, fuckass, fuck-ass, fuck-ass, fuckbag, fuck-bitch, fuck-bitch, fuckboy, fuckbrain, fuckbutt, fuckbutter, fucked, fuckedup, fucker, fuckers, fuckersucker, fuckface, fuckhead, fuckheads, fuckhole, fuckin, fucking, fuckings, fuckingshitmotherfucker, fuckme, fuckme, fuckmeat, fuckmeat, fucknugget, fucknut, fucknutt, fuckoff, fucks, fuckstick, fucktard, fuck-tard, fucktards, fucktart, fucktoy, fucktoy, fucktwat, fuckup, fuckwad, fuckwhit, fuckwit, fuckwitt, fudge packer, fudgepacker, fudge-packer, fuk, fuker, fukker, fukkers, fukkin, fuks, fukwhit, fukwit, fuq, futanari, fux, fux0r, fvck, fxck, gae, gai, gang bang, gangbang, gang-bang, gang-bang, gangbanged, gangbangs, ganja, gash, gassy ass, gassy ass, gay, gay sex, gayass, gaybob, gaydo, gayfuck, gayfuckist, gaylord, gays, gaysex, gaytard, gaywad, gender bender, genitals, gey, gfy, ghay, ghey, giant cock, gigolo, ginger, gippo, girl on, girl on top, girls gone wild, git, glans, goatcx, goatse, god, god damn, godamn, godamnit, goddam, god-dam, goddammit, goddamn, goddamned, god-damned, goddamnit, godsdamn, gokkun, golden shower, goldenshower, golliwog, gonad, gonads, goo girl, gooch, goodpoop, gook, gooks, goregasm, gringo, grope, group sex, gspot, g-spot, gtfo, guido, guro, h0m0, h0mo, ham flap, ham flap, hand job, handjob, hard core, hard on, hardcore, hardcoresex, he11, hebe, heeb, hell, hemp, hentai, heroin, herp, herpes, herpy, heshe, he-she, hircismus, hitler, hiv, ho, hoar, hoare, hobag, hoe, hoer, holy shit, hom0, homey, homo, homodumbshit, homoerotic, homoey, honkey, honky, hooch, hookah, hooker, hoor, hootch, hooter, hooters, hore, horniest, horny, hot carl, hot chick, hotsex, how to kill, how to murdep, how to murder, huge fat, hump, humped, humping, hun, hussy, hymen, iap, iberian slap, inbred, incest, injun, intercourse, jack off, jackass, jackasses, jackhole, jackoff, jack-off, jaggi, jagoff, jail bait, jailbait, jap, japs, jelly donut, jerk, jerk off, jerk0ff, jerkass, jerked, jerkoff, jerk-off, jigaboo, jiggaboo, jiggerboo, jism, jiz, jiz, jizm, jizm, jizz, jizzed, jock, juggs, jungle bunny, junglebunny, junkie, junky, kafir, kawk, kike, kikes, kill, kinbaku, kinkster, kinky, klan, knob, knob end, knobbing, knobead, knobed, knobend, knobhead, knobjocky, knobjokey, kock, kondum, kondums, kooch, kooches, kootch, kraut, kum, kummer, kumming, kums, kunilingus, kunja, kunt, kwif, kwif, kyke, l3i+ch, l3itch, labia, lameass, lardass, leather restraint, leather straight jacket, lech, lemon party, LEN, leper, lesbian, lesbians, lesbo, lesbos, lez, lezza/lesbo, lezzie, lmao, lmfao, loin, loins, lolita, looney, lovemaking, lube, lust, lusting, lusty, m0f0, m0fo, m45terbate, ma5terb8, ma5terbate, mafugly, mafugly, make me come, male squirting, mams, masochist, massa, masterb8, masterbat*, masterbat3, masterbate, master-bate, master-bate, masterbating, masterbation, masterbations, masturbate, masturbating, masturbation, maxi, mcfagget, menage a trois, menses, menstruate, menstruation, meth, m-fucking, mick, microphallus, middle finger, midget, milf, minge, minger, missionary position, mof0, mofo, mo-fo, molest, mong, moo moo foo foo, moolie, moron, mothafuck, mothafucka, mothafuckas, mothafuckaz, mothafucked, mothafucked, mothafucker, mothafuckers, mothafuckin, mothafucking, mothafucking, mothafuckings, mothafucks, mother fucker, mother fucker, motherfuck, motherfucka, motherfucked, motherfucker, motherfuckers, motherfuckin, motherfucking, motherfuckings, motherfuckka, motherfucks, mound of venus, mr hands, muff, muff diver, muff puff, muff puff, muffdiver, muffdiving, munging, munter, murder, mutha, muthafecker, muthafuckker, muther, mutherfucker, n1gga, n1gger, naked, nambla, napalm, nappy, nawashi, nazi, nazism, need the dick, need the dick, negro, neonazi, nig nog, nigaboo, nigg3r, nigg4h, nigga, niggah, niggas, niggaz, nigger, niggers, niggle, niglet, nig-nog, nimphomania, nimrod, ninny, ninnyhammer, nipple, nipples, nob, nob jokey, nobhead, nobjocky, nobjokey, nonce, nsfw images, nude, nudity, numbnuts, nut butter, nut butter, nut sack, nutsack, nutter, nympho, nymphomania, octopussy, old bag, omg, omorashi, one cup two girls, one guy one jar, opiate, opium, orally, organ, orgasim, orgasims, orgasm, orgasmic, orgasms, orgies, orgy, ovary, ovum, ovums, p.u.s.s.y., p0rn, paedophile, paki, panooch, pansy, pantie, panties, panty, pawn, pcp, pecker, peckerhead, pedo, pedobear, pedophile, pedophilia, pedophiliac, pee, peepee, pegging, penetrate, penetration, penial, penile, penis, penisbanger, penisfucker, penispuffer, perversion, phallic, phone sex, phonesex, phuck, phuk, phuked, phuking, phukked, phukking, phuks, phuq, piece of shit, pigfucker, pikey, pillowbiter, pimp, pimpis, pinko, piss, piss off, piss pig, pissed, pissed off, pisser, pissers, pisses, pisses, pissflaps, pissin, pissin, pissing, pissoff, pissoff, piss-off, pisspig, playboy, pleasure chest, pms, polack, pole smoker, polesmoker, pollock, ponyplay, poof, poon, poonani, poonany, poontang, poop, poop chute, poopchute, Poopuncher, porch monkey, porchmonkey, porn, porno, pornography, pornos, pot, potty, prick, pricks, prickteaser, prig, prince albert piercing, prod, pron, prostitute, prude, psycho, pthc, pube, pubes, pubic, pubis, punani, punanny, punany, punkass, punky, punta, puss, pusse, pussi, pussies, pussy, pussy fart, pussy fart, pussy palace, pussy palace, pussylicking, pussypounder, pussys, pust, puto, queaf, queaf, queef, queer, queerbait, queerhole, queero, queers, quicky, quim, racy, raghead, raging boner, rape, raped, raper, rapey, raping, rapist, raunch, rectal, rectum, rectus, reefer, reetard, reich, renob, retard, retarded, reverse cowgirl, revue, rimjaw, rimjob, rimming, ritard, rosy palm, rosy palm and her 5 sisters, rtard, r-tard, rubbish, rum, rump, rumprammer, ruski, rusty trombone, s hit, s&m, s.h.i.t., s.o.b., s_h_i_t, s0b, sadism, sadist, sambo, sand nigger, sandbar, sandbar, Sandler, sandnigger, sanger, santorum, sausage queen, sausage queen, scag, scantily, scat, schizo, schlong, scissoring, screw, screwed, screwing, scroat, scrog, scrot, scrote, scrotum, scrud, scum, seaman, seamen, seduce, seks, semen, sex, sexo, sexual, sexy, sh!+, sh!t, sh1t, s-h-1-t, shag, shagger, shaggin, shagging, shamedame, shaved beaver, shaved pussy, shemale, shi+, shibari, shirt lifter, shit, s-h-i-t, shit ass, shit fucker, shit fucker, shitass, shitbag, shitbagger, shitblimp, shitbrains, shitbreath, shitcanned, shitcunt, shitdick, shite, shiteater, shited, shitey, shitface, shitfaced, shitfuck, shitfull, shithead, shitheads, shithole, shithouse, shiting, shitings, shits, shitspitter, shitstain, shitt, shitted, shitter, shitters, shitters, shittier, shittiest, shitting, shittings, shitty, shiz, shiznit, shota, shrimping, sissy, skag, skank, skeet, skullfuck, slag, slanteye, slave, sleaze, sleazy, slope, slope, slut, slut bucket, slut bucket, slutbag, slutdumper, slutkiss, sluts, smartass, smartasses, smeg, smegma, smut, smutty, snatch, sniper, snowballing, snuff, s-o-b, sod off, sodom, sodomize, sodomy, son of a bitch, son of a motherless goat, son of a whore, son-of-a-bitch, souse, soused, spac, spade, sperm, spic, spick, spik, spiks, splooge, splooge moose, spooge, spook, spread legs, spunk, stfu, stiffy, stoned, strap on, strapon, strappado, strip, strip club, stroke, stupid, style doggy, suck, suckass, sucked, sucking, sucks, suicide girls, sultry women, sumofabiatch, swastika, swinger, t1t, t1tt1e5, t1tties, taff, taig, tainted love, taking the piss, tampon, tard, tart, taste my, tawdry, tea bagging, teabagging, teat, teets, teez, teste, testee, testes, testical, testicle, testis, threesome, throating, thrust, thug, thundercunt, tied up, tight white, tinkle, tit, tit wank, tit wank, titfuck, titi, tities, tits, titt, tittie5, tittiefucker, titties, titty, tittyfuck, tittyfucker, tittywank, titwank, toke, tongue in a, toots, topless, tosser, towelhead, tramp, tranny, transsexual, trashy, tribadism, trumped, tub girl, tubgirl, turd, tush, tushy, tw4t, twat, twathead, twatlips, twats, twatty, twatwaffle, twink, twinkie, two fingers, two fingers with tongue, two girls one cup, twunt, twunter, ugly, unclefucker, undies, undressing, unwed, upskirt, urethra play, urinal, urine, urophilia, uterus, uzi, v14gra, v1gra, vag, vagina, vajayjay, va-j-j, valium, venus mound, veqtable, viagra, vibrator, violet wand, virgin, vixen, vjayjay, vodka, vomit, vorarephilia, voyeur, vulgar, vulva, w00se, wad, wang, wank, wanker, wankjob, wanky, wazoo, wedgie, weed, weenie, weewee, weiner, weirdo, wench, wet dream, wetback, wh0re, wh0reface, white power, whiz, whoar, whoralicious, whore, whorealicious, whorebag, whored, whoreface, whorehopper, whorehouse, whores, whoring, wigger, willies, willy, window licker, wiseass, wiseasses, wog, womb, wop, wrapping men, wrinkled starfish, wtf, xrated, x-rated, xx, xxx, yaoi, yeasty, yellow showers, yid, yiffy, yobbo, zibbi, zoophilia, zubb"
bad_words = "2 girls 1 cup, 2g1c, 4r5e, 5h1t, 5hit, a$$"
bad_words = [x.strip() for x in bad_words.split(",")]

def clean_text(x):
    x = str(x)
    for word in bad_words:
        x = x.replace(word, "ZZZZZZ")
    for punct in puncts:
        # Add space after and before punct
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{10,}', '######', x)
    x = re.sub('[0-9]{5,10}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

# Clean speelings
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

# Not used because Custom packages are not supported for GPU instances
def split_text(x):
    # Probabilistically split concatenated words using NLP based on English Wikipedia uni-gram frequencies.
    x = wordninja.split(x)
    return '-'.join(x)

def add_features(df):
    
    df['question_text'] = df['question_text'].progress_apply(lambda x:str(x))
    df['total_length'] = df['question_text'].progress_apply(len)
    df['capitals'] = df['question_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.progress_apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].progress_apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']  

    return df

def load_and_prec():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    
    for df in [train_df, test_df]:
        df["question_text"] = df["question_text"].str.lower()
        df["question_text"] = df["question_text"].progress_apply(lambda x: clean_text(x))
        df["question_text"] = df["question_text"].progress_apply(lambda x: clean_numbers(x))
        df["question_text"] = df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
        df["question_text"] = df["question_text"].fillna("_##_")   ## Fill up the missing values
    
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ############## Add Features #############
    _train = add_features(train_df)
    _test = add_features(test_df)

    #features = _train[['caps_vs_length', 'words_vs_unique']].fillna(0)
    #test_features = _test[['caps_vs_length', 'words_vs_unique']].fillna(0)
    ff = ['total_length', 'capitals', 'caps_vs_length', 'num_words', 'num_unique_words', 'words_vs_unique']
    features = _train[ff].fillna(0)
    test_features = _test[ff].fillna(0)

    ss = StandardScaler()
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)
    
    ############## Split to train and val ##############
#     train_df, val_df = train_test_split(train_df, test_size=0.001, random_state=2018)
#     train_X = train_df["question_text"].values
#     val_X = val_df["question_text"].values
#     test_X = test_df["question_text"].values
    
    # Splitting to training and a final test set
    train_X, valid_X, train_y, val_y = train_test_split(
        list(zip(train_df['question_text'].values, features)),
        train_df['target'].values,
        test_size=0.2, random_state=2018
    )
    test_X = test_df["question_text"].values
    train_X, features = zip(*train_X)
    val_X, val_features = zip(*valid_X)

    ############## Tokenize the sentences ##############
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ############## Pad the sentences ##############
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    
#     ############## Shuffling the data ##############
#     np.random.seed(2018)
#     trn_idx = np.random.permutation(len(train_X))
#     val_idx = np.random.permutation(len(val_X))

#     train_X, val_X = train_X[trn_idx], val_X[val_idx] # .astype(int)
#     train_y, val_y = train_y[trn_idx], val_y[val_idx]
#     features, val_features = features[trn_idx], val_features[val_idx]
    
    return train_X, val_X, test_X, train_y, val_y, features, val_features, test_features, tokenizer.word_index
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]
    print(emb_mean,emb_std,"para")

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
# https://www.kaggle.com/yekenot/2dcnn-textclassifier
def model_cnn(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
def model_lstm_atten(embedding_matrix):
    inp1 = Input(shape=(maxlen,), name='inp1')
    inp2 = Input( shape=(nb_features,), name='inp2')
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp1)
    x = SpatialDropout1D(0.25)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = concatenate( [x, inp2] )
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[inp1, inp2], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def train_pred(model, epochs=2):
    for e in range(epochs):
        model.fit({'inp1': train_X, 'inp2': features}, train_y, batch_size=512, epochs=1,
                  validation_data=({'inp1': val_X, 'inp2': val_features}, val_y))
        pred_val_y = model.predict({'inp1': val_X, 'inp2': val_features}, batch_size=1024, verbose=0)
    pred_test_y = model.predict({'inp1': test_X, 'inp2': test_features}, batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y
train_X, val_X, test_X, train_y, val_y, features, val_features, test_features, word_index = load_and_prec()
########### SAVE DATASET TO DISK ############
np.save("train_X",train_X)
np.save("val_X",val_X)
np.save("test_X",test_X)
np.save("train_y",train_y)
np.save("val_y",val_y)

np.save("features",features)
np.save("val_features",val_features)
np.save("test_features",test_features)
np.save("word_index.npy",word_index)

######### LOAD DATASET FROM DISK ###########
train_X = np.load("train_X.npy")
val_X = np.load("val_X.npy")
test_X = np.load("test_X.npy")
train_y = np.load("train_y.npy")
val_y = np.load("val_y.npy")

features = np.load("features.npy")
val_features = np.load("val_features.npy")
test_features = np.load("test_features.npy")
word_index = np.load("word_index.npy").item()
embedding_matrix_1 = load_glove(word_index)
embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)
### Simple average: http://aclweb.org/anthology/N18-2031 
embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2, embedding_matrix_3], axis = 0)
np.shape(embedding_matrix)
outputs = []
pred_val_y, pred_test_y = train_pred(model_lstm_atten(embedding_matrix), epochs = EPOCHS)
outputs.append([pred_val_y, pred_test_y, '2 LSTM w/ attention'])
from sklearn.metrics import f1_score

score = 0
thresh = .5
for i in np.arange(0.1, 0.991, 0.01):
    y_val = (np.array(pred_val_y) > i).astype(np.int)
    temp_score = f1_score(val_y, y_val)
    if(temp_score > score):
        score = temp_score
        thresh = i

print("CV: {}, Threshold: {}".format(score, thresh))
# weights = [0.20, 0.079, 0.12, 0.14, 0.15, 0.17, 0.14]
# pred_test_y = np.sum([outputs[i][1] * weights[i] for i in range(len(outputs))], axis = 0)
pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis = 0)

pred_test_y = (pred_test_y > thresh).astype(int)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

############## Add Features #############
_train = add_features(train_df)
_test = add_features(test_df)

#features = _train[['caps_vs_length', 'words_vs_unique']].fillna(0)
#test_features = _test[['caps_vs_length', 'words_vs_unique']].fillna(0)
features = _train.fillna(0)
test_features = _test.fillna(0)

ss = StandardScaler()
ss.fit(np.vstack((features, test_features)))
features = ss.transform(features)
test_features = ss.transform(test_features)

# ############## Split to train and val ##############
# #     train_df, val_df = train_test_split(train_df, test_size=0.001, random_state=2018)
# #     train_X = train_df["question_text"].values
# #     val_X = val_df["question_text"].values
# #     test_X = test_df["question_text"].values

# # Splitting to training and a final test set
# train_X, valid_X, train_y, val_y = train_test_split(
#     list(zip(train_df['question_text'].values, features)),
#     train_df['target'].values,
#     test_size=0.2, random_state=2018
# )
# test_X = test_df["question_text"].values
# train_X, features = zip(*train_X)
# val_X, val_features = zip(*valid_X)

# ############## Tokenize the sentences ##############
# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(train_X))
# train_X = tokenizer.texts_to_sequences(train_X)
# val_X = tokenizer.texts_to_sequences(val_X)
# test_X = tokenizer.texts_to_sequences(test_X)

# ############## Pad the sentences ##############
# train_X = pad_sequences(train_X, maxlen=maxlen)
# val_X = pad_sequences(val_X, maxlen=maxlen)
# test_X = pad_sequences(test_X, maxlen=maxlen)
_train.shape
_test.shape
