from fastai.text import *

import pickle

import sklearn.model_selection
PATH = Path('/kaggle/working')
with open('../input/out.pickle', 'rb') as my_file:

    df = pickle.load(my_file)
trn_texts,val_texts = sklearn.model_selection.train_test_split(

    df, test_size=0.5)
df_trn = trn_texts

df_val = val_texts
data_lm = TextLMDataBunch.from_df(PATH, df_trn, df_val,tokenizer = Tokenizer())
data_clas = TextClasDataBunch.from_df(PATH, df_trn, df_val, vocab=data_lm.train_ds.vocab, bs=32)
data_lm.save('data_lm_export.pkl')

data_clas.save('data_clas_export.pkl')
data_lm = load_data(PATH, 'data_lm_export.pkl')

data_clas = load_data(PATH, 'data_clas_export.pkl', bs=16)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

learn.fit_one_cycle(1, 1e-2)
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc')
data_clas.show_batch()
learn.fit_one_cycle(1, 1e-2)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8, 1e-3)
learn.predict("Oh, Valentine’s Day, how I love thee! Though you are a holiday created by Hallmark, you give me an excuse to wear loads of pink and do all sorts of crafts involving hearts and other girly things, so for that, I thank you. (Also I get to celebrate how much I love my wonderful hubby, and I’ll look for any excuse to do that!) For some reason I woke up in a Valentines-y mood today, and I just needed to do something to celebrate. Enter, brownies. I sort of made them up as I went along, but they sure look like Valentine’s Day exploded all over them, so I’ll chalk them up as a success. Here’s what I used for this creation (Note: Not pictured are two eggs and vegetable oil, which are needed to complete the brownie mix):   Yes, I used a boxed mix. I wanted to keep it simple. Don’t hate. 😉 You’ll also want some heart-shaped cookie cutters.  I started by making the brownies according to the instructions on the package. I opted for a 13 x 9 sized pan so that the brownies would be thin enough to cut with a cookie cutter. Make sure you grease the pan really well. After they’re done baking, let them cool completely. I let them sit for about an hour and a half before I did anything else with them, but yours may not take that long.  After they’ve cooled, use the cookie cutters to make your heart shapes. Tip: Use a plastic knife to help reinforce the shape created by the cookie cutter, and run it underneath your brownie to make sure none of it sticks to the pan. For some reason brownies stick to plastic-ware way less than silverware.  I laid out my heart brownies on wax paper to apply the chocolate drizzle. I then melted slightly less than half the bag of the red chocolate pieces in a Pyrex measuring cup, microwaving them for about a minute at a time with the power level set at 5 and stirring after each minute. It took about five minutes to get them fully melted. (Make sure the chocolate is completely melted through; otherwise drizzling can get messy!)  I poured the liquified chocolate into the squeeze bottle and started drizzling away! I then repeated the process with the white chocolate. Tip: For the prettiest drizzly look, move your whole arm (as opposed to just your wrist) back and forth over the brownies while squeezing the bottle as you drizzle the melted chocolate.  I ended up using the candy hearts on a few of the brownies, but I also liked how they looked without them, so I left some of them just drizzled.  Ta-da! I love these brownies because they’re simple and beautiful. Don’t be intimidated if you’ve never worked with melted chocolate before– it is a breeze! This is a fun way to dress up everyday brownies and add that extra special “wow” factor. Enjoy!  What’s your favorite Valentine’s Day Treat? ")
learn.predict("'I know first hand how this economy is taking its toll.  Our family is trying hard to save and cut things out where we can in order to save a little here and there.  Lately, it seems like every time I turn around there is another birthday and Father’s Day is right around the corner!  One way we sometimes save a little is by sending Free Ecards!  Not only do we save on the cost of the card, but we save on the postage.  Another benefit is that you can select the day you would like the card sent so that it actually arrives on the person’s birthday or the designated holiday! Got Free Cards takes it a step further and makes them animated!  ")
learn.predict(" Ding-Dong. Who’s There? The Safety Police!  November 19, 2009      That’s what’s happening in England, folks: A new proposal to have safety experts go into families’ \xa0homes to make sure they’re utterly safe, right down the stair guards. Woe to the family that babyproofs in a manner not approved by the state! Here’s the Times OnLine article    about the new guidelines, and here’s a comment someone posted that I especially enjoyed: [The article said that] “About 100,000 children are admitted to hospital each year for home injuries at a cost of £146m.” I’m delighted to hear that. It indicates that at least 100,000 of the next generation will still be willing to take risks and behave uninhibitedly. It’s the other 10,000,000 I worry about, the ones who will be taught never to go out in the rain in case they catch a cold.  Are you kidding? Safety is GOOD. I personally love window guards and I put latches on my cabinets when my kids were younger. But I, for instance, think toilet locks are a waste of money. What if the government disagrees? What if I think it’s my job to teach the kids not to open the oven, but the government believes I ought to invest in some oven guards? And what if the government ends up endorsing baby knee pads? (Maybe that’s good because that way, when I feel like banging my head against a baby’s knee in utter frustration, neither of us will get hurt.) \xa0— Lenore      England, home, safety")