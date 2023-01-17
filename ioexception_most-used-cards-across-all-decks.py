import json
from pathlib import Path
import csv
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib.offsetbox import AnchoredText
# https://fontsup.com/es/font/matrix-regular-small-caps.html
!wget -O card_name.ttf https://www.dropbox.com/s/tfwt68u14ncm2zo/MatrixRegularSmallCaps.ttf?dl=1
# https://fontsup.com/es/font/matrix-bold-small-caps.html
!wget -O card_number.ttf https://www.dropbox.com/s/gil5eef4x900nao/MatrixBoldSmallCaps_29827.ttf?dl=1
# https://fontsup.com/es/font/matrix-book.html
!wget -O card_effect.ttf https://www.dropbox.com/s/0psxpekkklsjuw8/Matrix-Book.ttf?dl=1
main_deck_card_counter = Counter()
for jsonl in sorted(Path("/kaggle/input").glob("**/*.jsonl")):
    for line in jsonl.open():
        deck_info = json.loads(line)
        if "deck" in deck_info and "main" in deck_info["deck"]:
            main_deck = set(deck_info["deck"]["main"])
            main_deck_card_counter.update(main_deck)
    
card_by_id = dict()
with open("/kaggle/input/yugioh-cards/cards.csv") as fp:
    reader = csv.DictReader(fp)
    for card in reader:
        card_by_id[card["id"]] = card
most_common = main_deck_card_counter.most_common(10)
cards, counts = list(zip(*most_common))
cards, counts = list(cards), list(counts)
def find_colors(card_type):
    if "Monster" in card_type:
        return "#FF8B53", "black"
    if "Trap" in card_type:
        return "#BC5A84", "white"
    if "Spell" in card_type:
        return "#1D9E74", "white"
fig = plt.figure(figsize=(15,10))
fig.patch.set_facecolor('white')
ax = fig.gca()
sns.barplot(x=counts, y=[card_by_id[card_id]["name"] for card_id in cards], ax=ax, orient="h")
ax.set_title("Most used cards in Yu-Gi-Oh! decks", size=20)
fig = plt.figure(figsize=(15,10))
fig.patch.set_facecolor('white')
ax = fig.gca()
sns.barplot(x=counts, y=[placeholder for placeholder in range(len(counts))], ax=ax, orient="h")

xlim = ax.get_xlim()
ratio = xlim[1] / 100

card_name_prop = fm.FontProperties(fname="card_name.ttf", size=40)
card_effect_prop = fm.FontProperties(fname="card_effect.ttf", size=40)
card_effect_prop_sm = fm.FontProperties(fname="card_effect.ttf", size=20)
card_number_prop = fm.FontProperties(fname="card_number.ttf", size=30)

for card_id, rect in zip(cards, ax.patches):
    x, y = rect.xy
    if card_id == "-1":
        continue
    card = card_by_id[card_id]
    rect_color, font_color = find_colors(card["type"])
    rect.set_facecolor(rect_color)
    ax.text(x + ratio , y + 0.6, card["name"], color=font_color, fontproperties=card_name_prop)

for label in ax.get_xticklabels() :
    label.set_fontproperties(card_number_prop)

text = AnchoredText("Data from decks built at ygoprodeck.com", loc=4,
                    prop={'size': 10}, frameon=True)
ax.add_artist(text)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel("number of decks", fontproperties=card_number_prop)
ax.set_title("Most used cards in Yu-Gi-Oh! decks", fontproperties=card_effect_prop)
pass
