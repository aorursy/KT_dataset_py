#imports
import pandas as pd
from ipywidgets import FloatProgress
from IPython.display import display
#Energy in KJ, everything else in g per 100g
dietary_cols=['Name','Moisture','Protein','Ash','Total Fat','Dietary Fibre','Carbohydrate','Energy']

cereals_and_millets=pd.DataFrame(columns=dietary_cols)
cereals_and_millets.loc[len(cereals_and_millets)]=['Amaranth seed, black (Amaranthus cruentus)',9.89, 14.59, 2.78, 5.74, 7.02, 59.98, 1490]
cereals_and_millets.loc[len(cereals_and_millets)]=['Amaranth seed, pale brown (Amaranthus cruentus)',9.20, 13.27, 3.05, 5.5, 7.47, 61.46, 1489]
cereals_and_millets.loc[len(cereals_and_millets)]=['Bajra (Pennisetum typhoideum)', 8.97, 10.96, 1.37, 5.43, 11.49, 61.78, 1456]
cereals_and_millets.loc[len(cereals_and_millets)]=['Barley (Hordeum vulgare)',9.77, 10.94, 1.06, 1.30, 15.64, 61.29, 1321]
cereals_and_millets.loc[len(cereals_and_millets)]=['Jowar (Sorghum vulgare) ', 9.0, 9.9, 1.3, 1.7, 10.2, 67.6, 1398.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Maize, dry (Zea mays) ', 9.2, 8.8, 1.1, 3.7, 12.2, 64.7, 1398.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Maize, tender, local (Zea mays) ', 68.2, 3.5, 0.3, 1.4, 3.6, 22.6, 502.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Maize, tender, sweet (Zea mays) ', 74.4, 4.1, 0.3, 1.3, 3.3, 16.4, 405.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Quinoa (Chenopodium quinoa) ', 10.4, 13.1, 2.6, 5.5, 14.6, 53.6, 1374.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Ragi (Eleusine coracana) ', 10.8, 7.1, 2.0, 1.9, 11.1, 66.8, 1342.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Rice flakes (Oryza sativa ) ', 10.3, 7.4, 0.8, 1.1, 3.4, 76.7, 1480.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Rice puffed (Oryza sativa ) ', 9.4, 7.4, 1.2, 1.6, 2.5, 77.6, 1514.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Rice, raw, brown (Oryza sativa ) ', 9.3, 9.1, 1.0, 1.2, 4.4, 74.8, 1480.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Rice, parboiled, milled (Oryza sativa ) ', 10.0, 7.8, 0.6, 0.5, 3.7, 77.1, 1471.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Rice, raw, milled (Oryza sativa ) ', 9.9, 7.9, 0.5, 0.5, 2.8, 78.2, 1491.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Samai (Panicum miliare) ', 11.3, 10.1, 1.3, 3.8, 7.7, 65.5, 1449.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Varagu (Setaria italica) ', 14.2, 8.9, 1.7, 2.5, 6.3, 66.1, 1388.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Wheat flour, refined (Triticum aestivum) ', 11.3, 10.3, 0.5, 0.7, 2.7, 74.2, 1472.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Wheat flour, atta (Triticum aestivum) ', 11.1, 10.5, 1.2, 1.5, 11.3, 64.1, 1340.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Wheat, whole (Triticum aestivum) ', 10.5, 10.5, 1.4, 1.4, 11.2, 64.7, 1347.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Wheat, bulgur (Triticum aestivum) ', 8.6, 10.8, 1.2, 1.4, 8.8, 69.0, 1430.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Wheat, semolina (Triticum aestivum) ', 8.9, 11.3, 0.8, 0.7, 9.7, 68.4, 1396.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Wheat, vermicelli (Triticum aestivum) ', 9.5, 9.7, 0.6, 0.4, 9.2, 70.3, 1392.0]
cereals_and_millets.loc[len(cereals_and_millets)]=['Wheat, vermicelli, roasted (Triticum aestivum) ', 7.6, 10.3, 0.5, 0.4, 9.5, 71.4, 1423.0]

print(cereals_and_millets.head())
grains=pd.DataFrame(columns=dietary_cols)
grains.loc[len(grains)]=['Bengal gram, dal (Cicer arietinum) ', 9.1, 21.5, 2.1, 5.3, 15.1, 46.7, 1377.0]
grains.loc[len(grains)]=['Bengal gram, whole (Cicer arietinum) ', 8.5, 18.7, 2.7, 5.1, 25.2, 39.5, 1201.0]
grains.loc[len(grains)]=['Black gram, dal (Phaseolus mungo) ', 9.1, 23.0, 3.1, 1.6, 11.9, 51.0, 1356.0]
grains.loc[len(grains)]=['Black gram, whole (Phaseolus mungo) ', 8.7, 21.9, 3.3, 1.5, 20.4, 43.9, 1219.0]
grains.loc[len(grains)]=['Cowpea, brown (Vigna catjang) ', 9.4, 20.3, 2.9, 1.1, 11.5, 54.6, 1340.0]
grains.loc[len(grains)]=['Cowpea, white (Vigna catjang) ', 9.3, 21.2, 2.8, 1.1, 11.7, 53.7, 1340.0]
grains.loc[len(grains)]=['Field bean, black (Phaseolus vulgaris) ', 9.5, 19.9, 2.7, 0.9, 23.4, 43.4, 1155.0]
grains.loc[len(grains)]=['Field bean, brown (Phaseolus vulgaris) ', 8.7, 19.9, 2.7, 0.9, 22.4, 45.2, 1184.0]
grains.loc[len(grains)]=['Field bean, white (Phaseolus vulgaris) ', 8.6, 19.8, 3.0, 0.9, 22.9, 44.5, 1173.0]
grains.loc[len(grains)]=['Green gram, dal (Phaseolus aureus) ', 9.7, 23.8, 3.0, 1.3, 9.3, 52.5, 1363.0]
grains.loc[len(grains)]=['Green gram, whole (Phaseolus aureus) ', 9.9, 22.5, 3.2, 1.1, 17.0, 46.1, 1229.0]
grains.loc[len(grains)]=['Horse gram, whole (Dolicus biflorus) ', 9.2, 21.7, 3.2, 0.6, 7.8, 57.2, 1379.0]
grains.loc[len(grains)]=['Lentil dal (Lens culinaris) ', 9.7, 24.3, 2.2, 0.7, 10.4, 52.5, 1349.0]
grains.loc[len(grains)]=['Lentil whole, brown (Lens culinaris) ', 9.2, 22.4, 2.3, 0.6, 16.8, 48.4, 1251.0]
grains.loc[len(grains)]=['Lentil whole, yellowish (Lens culinaris) ', 9.7, 22.8, 2.2, 0.6, 16.6, 47.9, 1246.0]
grains.loc[len(grains)]=['Moth bean (Vigna aconitifolia) ', 8.1, 19.7, 3.1, 1.7, 15.1, 52.0, 1291.0]
grains.loc[len(grains)]=['Peas, dry (Pisum sativum) ', 9.3, 20.4, 2.4, 1.8, 17.0, 48.9, 1269.0]
grains.loc[len(grains)]=['Rajmah, black (Phaseolus vulgaris) ', 8.6, 19.0, 3.3, 1.6, 17.7, 49.5, 1247.0]
grains.loc[len(grains)]=['Rajmah, brown (Phaseolus vulgaris) ', 9.6, 19.5, 3.3, 1.6, 16.9, 48.8, 1245.0]
grains.loc[len(grains)]=['Rajmah, red (Phaseolus vulgaris) ', 9.8, 19.9, 3.2, 1.7, 16.5, 48.6, 1252.0]
grains.loc[len(grains)]=['Red gram, dal (Cajanus cajan) ', 9.2, 21.7, 3.2, 1.5, 9.0, 55.2, 1384.0]
grains.loc[len(grains)]=['Red gram, whole (Cajanus cajan) ', 9.3, 20.4, 3.5, 1.3, 22.8, 42.4, 1146.0]
grains.loc[len(grains)]=['Ricebean (Vigna umbellata ) ', 11.1, 19.9, 3.5, 0.7, 13.3, 51.2, 1265.0]
grains.loc[len(grains)]=['Soya bean, brown (Glycine max) ', 5.5, 35.5, 4.7, 19.8, 21.5, 12.7, 1596.0]
grains.loc[len(grains)]=['Soya bean, white (Glycine max) ', 5.4, 37.8, 4.5, 19.4, 22.6, 10.1, 1579.0]

print(grains.head())
green_leaf_veg=pd.DataFrame(columns=dietary_cols)
green_leaf_veg.loc[len(green_leaf_veg)]=['Agathi leaves (Sesbania grandiflora) ', 74.4, 8.0, 2.4, 1.3, 8.6, 5.2, 295.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Amaranth leaves, green (Amaranthus gangeticus) ', 86.8, 3.2, 2.5, 0.6, 4.4, 2.2, 1.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Amaranth leaves, red (Amaranthus gangeticus) ', 85.5, 3.9, 2.6, 0.6, 4.9, 2.3, 140.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Amaranth leaves, red and green mix(Amaranthus gangeticus) ', 86.3, 3.0, 2.5, 0.5, 4.6, 2.8, 132.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Amaranth spinosus, leaves, green (Amaranthus spinosus) ', 86.4, 3.5, 2.9, 0.3, 5.1, 1.6, 110.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Amaranth spinosus, leaves, red and green mix (Amaranthus spinosus) ', 86.6, 2.8, 3.2, 0.3, 5.5, 1.4, 99.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Basella leaves (Basella alba) ', 92.6, 1.5, 1.0, 0.4, 2.2, 2.0, 82.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Bathua leaves (Chenopodium album) ', 88.7, 2.5, 1.7, 0.4, 4.0, 2.5, 116.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Beet greens (Beta vulgaris) ', 86.6, 2.3, 2.6, 0.7, 3.6, 3.8, 145.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Betel leaves, big (kolkata) (Piper betle) ', 84.9, 2.5, 2.3, 0.7, 2.1, 7.3, 202.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Betel leaves, small (Piper betle) ', 85.9, 2.6, 2.5, 0.7, 1.9, 6.1, 183.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Brussels sprouts (Brassica oleracea var. gemmifera) ', 84.3, 4.2, 1.4, 0.5, 4.2, 5.0, 185.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Cabbage, Chinese (Brassica rupa) ', 93.1, 1.5, 0.7, 0.1, 2.0, 2.3, 75.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Cabbage, collard greens (Brassica oleracea var.viridis) ', 89.5, 3.6, 0.8, 0.2, 2.9, 2.7, 126.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Cabbage, green (Brassica oleracea var. capitata f. alba) ', 91.8, 1.3, 0.6, 0.1, 2.7, 3.2, 90.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Cabbage, violet (Brassica oleracea var. capitata f. rubra) ', 91.9, 1.3, 0.7, 0.2, 2.2, 3.5, 97.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Cauliflower leaves (Brassica oleracea var. botrytis) ', 87.6, 3.9, 1.2, 0.4, 3.4, 3.3, 148.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Colocasia leaves, green (Colocasia anti-quorum) ', 83.6, 3.4, 2.3, 1.3, 5.6, 3.6, 182.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Drumstick leaves (Moringa oleifera) ', 75.6, 6.4, 2.4, 1.6, 8.2, 5.6, 282.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Fenugreek leaves (Trigonella foenum graecum) ', 86.7, 3.6, 1.6, 0.8, 4.9, 2.1, 144.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Garden cress (Lepidium sativum) ', 84.0, 5.6, 2.4, 0.8, 2.6, 4.4, 208.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Gogu leaves, green (Hibiscus cannabinus) ', 87.4, 1.8, 0.9, 1.0, 4.5, 4.0, 152.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Gogu leaves, red (Hibiscus cannabinus) ', 87.9, 1.8, 0.9, 1.0, 3.8, 4.2, 153.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Knol-Khol, leaves (Brassica oleracea var. gongylodes) ', 86.2, 3.1, 1.4, 0.3, 2.7, 6.1, 178.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Lettuce (Lactuca sativa) ', 92.2, 1.5, 1.1, 0.2, 1.7, 3.0, 91.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Mustard leaves (Brassica juncea) ', 88.1, 3.5, 1.4, 0.5, 3.9, 2.4, 127.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Pak Choi leaves (Brassica rapa var. Chinensis) ', 93.5, 1.4, 1.1, 0.2, 1.9, 1.7, 67.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Parsley (Petroselinum crispum) ', 77.7, 5.5, 2.2, 1.1, 3.8, 9.4, 305.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Ponnaganni (Alternanthera sessilis) ', 79.4, 5.2, 2.6, 0.7, 6.7, 5.1, 213.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Pumpkin leaves, tender (Cucurbita maxima) ', 85.8, 4.2, 2.2, 0.7, 2.2, 4.7, 185.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Radish leaves (Raphanus sativus) ', 91.1, 2.2, 1.5, 0.5, 1.8, 2.7, 109.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Rumex leaves (Rumex patientia) ', 93.1, 1.6, 1.2, 0.3, 1.2, 2.3, 82.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Spinach (Spinacia oleracea) ', 90.3, 2.1, 2.4, 0.6, 2.3, 2.0, 102.0]
green_leaf_veg.loc[len(green_leaf_veg)]=['Tamarind leaves, tender (Tamarindus indica) ', 71.6, 5.8, 1.2, 0.4, 10.7, 10.0, 299.0]

print(green_leaf_veg.head())
other_veg=pd.DataFrame(columns=dietary_cols)
other_veg.loc[len(other_veg)]=['Ash gourd (Benincasa hispida) ', 92.1, 0.7, 0.7, 0.1, 3.3, 2.8, 73.0]
other_veg.loc[len(other_veg)]=['Bamboo shoot, tender (Bambusa vulgaris) ', 94.5, 1.3, 0.5, 0.3, 1.5, 1.6, 68.0]
other_veg.loc[len(other_veg)]=['Bean scarlet, tender (Phaseolus coccineus) ', 85.5, 2.8, 0.9, 0.9, 4.5, 5.1, 179.0]
other_veg.loc[len(other_veg)]=['Bitter gourd, jagged, teeth ridges, elongate (Momordica charantia) ', 90.8, 1.4, 0.8, 0.2, 3.7, 2.8, 87.0]
other_veg.loc[len(other_veg)]=['Bitter gourd, jagged, teeth ridges, short (Momordica charantia) ', 91.6, 1.3, 0.8, 0.2, 3.4, 2.5, 79.0]
other_veg.loc[len(other_veg)]=['Bitter gourd, jagged, smooth ridges, elongate (Momordica charantia) ', 91.2, 1.6, 0.8, 0.2, 3.7, 2.2, 81.0]
other_veg.loc[len(other_veg)]=['Bottle gourd, elongate, pale green (Lagenaria vulgaris) ', 95.1, 0.5, 0.3, 0.1, 2.1, 1.6, 46.0]
other_veg.loc[len(other_veg)]=['Bottle gourd, round, pale green (Lagenaria vulgaris) ', 94.5, 0.4, 0.3, 0.1, 2.1, 2.5, 57.0]
other_veg.loc[len(other_veg)]=['Bottle gourd, elongate, dark green (Lagenaria vulgaris) ', 94.6, 0.4, 0.4, 0.1, 2.1, 2.2, 54.0]
other_veg.loc[len(other_veg)]=['Brinjal-1 (Solanum melongena) ', 89.9, 1.7, 0.8, 0.3, 3.5, 3.4, 114.0]
other_veg.loc[len(other_veg)]=['Brinjal-2 (Solanum melongena) ', 90.2, 1.8, 0.8, 0.3, 4.0, 2.7, 99.0]
other_veg.loc[len(other_veg)]=['Brinjal-3 (Solanum melongena) ', 90.0, 1.3, 0.6, 0.3, 4.2, 3.3, 102.0]
other_veg.loc[len(other_veg)]=['Brinjal-4 (Solanum melongena) ', 90.2, 1.5, 0.6, 0.3, 4.0, 3.1, 100.0]
other_veg.loc[len(other_veg)]=['Brinjal-5 (Solanum melongena) ', 89.4, 1.3, 0.7, 0.2, 4.1, 4.0, 111.0]
other_veg.loc[len(other_veg)]=['Brinjal-6 (Solanum melongena) ', 90.5, 1.4, 0.7, 0.2, 3.6, 3.2, 97.0]
other_veg.loc[len(other_veg)]=['Brinjal-7 (Solanum melongena) ', 91.2, 1.4, 0.5, 0.3, 3.3, 3.1, 98.0]
other_veg.loc[len(other_veg)]=['Brinjal-8 (Solanum melongena) ', 89.2, 1.8, 0.8, 0.3, 4.0, 3.7, 117.0]
other_veg.loc[len(other_veg)]=['Brinjal-9 (Solanum melongena) ', 89.8, 1.4, 0.8, 0.3, 4.0, 3.5, 106.0]
other_veg.loc[len(other_veg)]=['Brinjal-10 (Solanum melongena) ', 89.3, 1.6, 0.7, 0.2, 3.9, 4.0, 116.0]
other_veg.loc[len(other_veg)]=['Brinjal-11 (Solanum melongena) ', 89.9, 1.4, 0.6, 0.3, 4.1, 3.5, 105.0]
other_veg.loc[len(other_veg)]=['Brinjal-12 (Solanum melongena) ', 90.9, 1.6, 0.5, 0.2, 3.8, 2.7, 93.0]
other_veg.loc[len(other_veg)]=['Brinjal-13 (Solanum melongena) ', 89.1, 1.4, 0.6, 0.3, 3.9, 4.5, 124.0]
other_veg.loc[len(other_veg)]=['Brinjal-14 (Solanum melongena) ', 90.3, 1.5, 0.6, 0.3, 3.8, 3.2, 106.0]
other_veg.loc[len(other_veg)]=['Brinjal-15 (Solanum melongena) ', 89.4, 1.5, 0.7, 0.2, 3.9, 3.9, 114.0]
other_veg.loc[len(other_veg)]=['Brinjal-16 (Solanum melongena) ', 90.3, 1.2, 0.8, 0.3, 3.7, 3.4, 103.0]
other_veg.loc[len(other_veg)]=['Brinjal-17 (Solanum melongena) ', 90.8, 1.1, 0.8, 0.3, 3.9, 2.8, 91.0]
other_veg.loc[len(other_veg)]=['Brinjal-18 (Solanum melongena) ', 89.0, 1.4, 0.6, 0.3, 4.3, 4.1, 116.0]
other_veg.loc[len(other_veg)]=['Brinjal-19 (Solanum melongena) ', 90.4, 1.2, 0.6, 0.3, 3.9, 3.3, 100.0]
other_veg.loc[len(other_veg)]=['Brinjal-20 (Solanum melongena) ', 90.6, 1.4, 0.6, 0.3, 3.7, 3.1, 99.0]
other_veg.loc[len(other_veg)]=['Brinjal-21 (Solanum melongena) ', 89.8, 1.3, 0.6, 0.3, 3.8, 3.9, 113.0]
other_veg.loc[len(other_veg)]=['Brinjal - all varieties (Solanum melongena) ', 90.0, 1.4, 0.7, 0.3, 3.9, 3.5, 106.0]
other_veg.loc[len(other_veg)]=['Broad beans (Vicia faba) ', 84.2, 3.8, 1.0, 0.1, 8.6, 2.1, 123.0]
other_veg.loc[len(other_veg)]=['Capsicum, green (Capsicum annuum) ', 93.8, 1.1, 0.7, 0.3, 2.0, 1.8, 68.0]
other_veg.loc[len(other_veg)]=['Capsicum, red (Capsicum annuum) ', 92.9, 1.4, 0.7, 0.4, 2.1, 2.1, 83.0]
other_veg.loc[len(other_veg)]=['Capsicum, yellow (Capsicum annuum) ', 93.3, 1.3, 0.7, 0.4, 2.1, 1.9, 78.0]
other_veg.loc[len(other_veg)]=['Cauliflower (Brassica oleracea) ', 90.7, 2.1, 0.9, 0.4, 3.7, 2.0, 96.0]
other_veg.loc[len(other_veg)]=['Celery stalk (Apium graveolens) ', 92.8, 0.9, 1.5, 0.2, 2.0, 2.3, 69.0]
other_veg.loc[len(other_veg)]=['Cho-cho-marrow (Sechium edule) ', 93.7, 0.6, 0.3, 0.1, 1.5, 3.4, 79.0]
other_veg.loc[len(other_veg)]=['Cluster beans (Cyamopsis tetragonobola) ', 84.6, 3.5, 1.6, 0.3, 4.8, 4.9, 168.0]
other_veg.loc[len(other_veg)]=['Colocasia, stem, black (Colocasia antiquorum) ', 91.1, 0.7, 0.9, 0.3, 3.0, 3.8, 100.0]
other_veg.loc[len(other_veg)]=['Colocasia, stem, green (Colocasia antiquorum) ', 92.6, 0.9, 1.0, 0.2, 2.3, 2.8, 81.0]
other_veg.loc[len(other_veg)]=['Corn, Baby (Zea mays) ', 75.4, 2.6, 2.7, 1.3, 6.0, 11.6, 306.0]
other_veg.loc[len(other_veg)]=['Cucumber, green, elongate (Cucumis sativus) ', 92.9, 0.7, 0.5, 0.1, 2.1, 3.4, 82.0]
other_veg.loc[len(other_veg)]=['Cucumber, green, short (Cucumis sativus) ', 93.5, 0.8, 0.5, 0.1, 2.1, 2.8, 73.0]
other_veg.loc[len(other_veg)]=['Cucumber, orange, round (Cucumis sativus) ', 92.8, 0.9, 0.5, 0.2, 2.4, 3.0, 82.0]
other_veg.loc[len(other_veg)]=['Drumstick (Moringa oleifera) ', 85.3, 2.6, 1.2, 0.1, 6.8, 3.7, 123.0]
other_veg.loc[len(other_veg)]=['Field beans, tender, broad (Vicia faba) ', 86.9, 3.0, 0.9, 0.6, 5.6, 2.7, 129.0]
other_veg.loc[len(other_veg)]=['Field beans, tender, lean (Vicia faba) ', 85.5, 3.7, 1.0, 0.6, 6.1, 2.8, 140.0]
other_veg.loc[len(other_veg)]=['French beans, country (Phaseolus vulgaris) ', 89.1, 2.4, 1.0, 0.2, 4.3, 2.6, 102.0]
other_veg.loc[len(other_veg)]=['French beans, hybrid (Phaseolus vulgaris) ', 90.1, 2.1, 0.7, 0.1, 4.1, 2.6, 93.0]
other_veg.loc[len(other_veg)]=['Jack fruit, raw (Artocarpus heterophyllus) ', 85.5, 1.9, 0.9, 0.3, 7.6, 3.4, 110.0]
other_veg.loc[len(other_veg)]=['Jack fruit, seed, mature (Artocarpus heterophyllus) ', 72.3, 5.7, 1.0, 0.4, 8.6, 11.8, 322.0]
other_veg.loc[len(other_veg)]=['Knol - Khol (Brassica oleracea) ', 93.1, 1.5, 0.7, 0.3, 2.7, 1.3, 67.0]
other_veg.loc[len(other_veg)]=['Kovai, big (Coccinia cordifolia) ', 92.7, 1.3, 0.5, 0.2, 3.0, 2.0, 73.0]
other_veg.loc[len(other_veg)]=['Kovai, small (Coccinia cordifolia) ', 92.4, 1.2, 0.4, 0.2, 3.2, 2.4, 80.0]
other_veg.loc[len(other_veg)]=['Ladies finger (Abelmoschus esculentus) ', 89.0, 2.0, 0.9, 0.2, 4.0, 3.6, 115.0]
other_veg.loc[len(other_veg)]=['Mango, green, raw (Mangifera indica) ', 85.1, 0.6, 0.4, 0.0, 3.0, 10.5, 205.0]
other_veg.loc[len(other_veg)]=['Onion, stalk (Allium cepa) ', 88.3, 2.0, 1.1, 0.2, 5.2, 2.9, 107.0]
other_veg.loc[len(other_veg)]=['Papaya, raw (Carica papaya) ', 92.0, 0.5, 0.5, 0.2, 2.2, 4.4, 100.0]
other_veg.loc[len(other_veg)]=['Parwar (Trichosanthes dioica) ', 91.5, 1.4, 0.5, 0.3, 2.6, 3.5, 101.0]
other_veg.loc[len(other_veg)]=['Peas, fresh (Pisum sativum) ', 73.3, 7.2, 1.0, 0.1, 6.3, 11.8, 340.0]
other_veg.loc[len(other_veg)]=['Plantain, flower (Musa x paradisiaca) ', 89.1, 1.4, 1.3, 0.6, 5.2, 2.1, 89.0]
other_veg.loc[len(other_veg)]=['Plantain, green (Musa x paradisiaca) ', 76.1, 1.1, 1.2, 0.2, 3.6, 17.5, 334.0]
other_veg.loc[len(other_veg)]=['Plantain, stem (Musa x paradisiaca) ', 87.5, 0.3, 1.2, 0.1, 2.1, 8.6, 165.0]
other_veg.loc[len(other_veg)]=['Pumpkin, green, cylindrical (Cucurbita maxima) ', 91.7, 0.8, 0.4, 0.1, 2.5, 4.2, 103.0]
other_veg.loc[len(other_veg)]=['Pumpkin, orange, round (Cucurbita maxima) ', 91.8, 0.8, 0.5, 0.1, 1.1, 4.0, 97.0]
other_veg.loc[len(other_veg)]=['Red gram, tender, fresh (Cajanus cajan) ', 64.0, 8.0, 1.6, 0.9, 5.9, 19.4, 520.0]
other_veg.loc[len(other_veg)]=['Ridge gourd (Luffa acutangula) ', 94.9, 0.9, 0.4, 0.1, 1.8, 1.7, 55.0]
other_veg.loc[len(other_veg)]=['Ridge gourd, smooth skin (Luffa acutangula) ', 94.2, 0.9, 0.5, 0.1, 1.8, 2.2, 64.0]
other_veg.loc[len(other_veg)]=['Snake gourd, long, pale green (Trichosanthes anguina) ', 94.8, 0.9, 0.4, 0.2, 2.2, 1.2, 52.0]
other_veg.loc[len(other_veg)]=['Snake gourd, long, dark green (Trichosanthes anguina) ', 94.9, 0.8, 0.4, 0.2, 0.5, 1.2, 50.0]
other_veg.loc[len(other_veg)]=['Snake gourd, short (Trichosanthes anguina) ', 94.3, 0.5, 0.4, 0.2, 2.2, 2.1, 61.0]
other_veg.loc[len(other_veg)]=['Tinda, tender (Praecitrullus fistulosus) ', 94.4, 1.0, 0.5, 0.1, 2.0, 1.9, 58.0]
other_veg.loc[len(other_veg)]=['Tomato, green (Lycopersicon esculentum) ', 93.2, 1.1, 0.6, 0.2, 1.6, 3.1, 87.0]
other_veg.loc[len(other_veg)]=['Tomato, ripe, hybrid (Lycopersicon esculentum) ', 93.7, 0.7, 0.4, 0.2, 1.5, 3.2, 79.0]
other_veg.loc[len(other_veg)]=['Tomato, ripe, local (Lycopersicon esculentum) ', 93.6, 0.9, 0.5, 0.4, 1.7, 2.7, 82.0]
other_veg.loc[len(other_veg)]=['Zucchini, green (Cucurbita pepo) ', 92.8, 1.1, 0.9, 0.5, 2.3, 2.3, 84.0]
other_veg.loc[len(other_veg)]=['Zucchini, yellow (Cucurbita pepo) ', 93.1, 1.3, 1.0, 0.4, 1.8, 2.2, 79.0]

print(other_veg.head())
fruits=pd.DataFrame(columns=dietary_cols)
fruits.loc[len(fruits)]=['Apple, big (Malus domestica) ', 83.0, 0.2, 0.3, 0.6, 2.5, 13.1, 261.0]
fruits.loc[len(fruits)]=['Apple, green (Malus domestica) ', 85.5, 0.4, 0.3, 0.5, 2.5, 10.6, 214.0]
fruits.loc[len(fruits)]=['Apple, small (Malus domestica) ', 82.9, 0.3, 0.2, 0.5, 2.0, 13.9, 267.0]
fruits.loc[len(fruits)]=['Apple, small, Kashmir (Malus sylvestris) ', 82.7, 0.2, 0.2, 0.6, 2.0, 13.9, 269.0]
fruits.loc[len(fruits)]=['Apricot, dried (Prunus armeniaca) ', 16.6, 3.1, 3.4, 0.7, 3.3, 72.6, 1321.0]
fruits.loc[len(fruits)]=['Apricot, processed (Prunus armeniaca) ', 85.7, 1.4, 0.6, 0.6, 0.5, 10.9, 236.0]
fruits.loc[len(fruits)]=['Avocado fruit (Persea americana) ', 73.5, 2.9, 1.1, 13.8, 6.6, 1.7, 604.0]
fruits.loc[len(fruits)]=['Bael fruit (Aegle marmelos) ', 61.3, 2.6, 0.9, 0.5, 6.3, 28.2, 569.0]
fruits.loc[len(fruits)]=['Banana, ripe, montham (Musa x paradisiaca) ', 70.1, 1.2, 1.1, 0.3, 2.2, 24.9, 463.0]
fruits.loc[len(fruits)]=['Banana, ripe, poovam (Musa x paradisiaca) ', 71.3, 1.4, 1.0, 0.3, 2.3, 23.4, 445.0]
fruits.loc[len(fruits)]=['Banana, ripe, red (Musa x paradisiaca) ', 70.2, 1.2, 0.9, 0.2, 1.9, 25.2, 467.0]
fruits.loc[len(fruits)]=['Banana, ripe, robusta (Musa x paradisiaca) ', 71.9, 1.2, 0.9, 0.3, 1.9, 23.6, 440.0]
fruits.loc[len(fruits)]=['Black berry (Rubus fruticosus) ', 82.9, 0.9, 0.5, 0.6, 4.3, 10.6, 227.0]
fruits.loc[len(fruits)]=['Cherries, red (Prunus cerasus) ', 83.6, 1.4, 0.4, 0.4, 2.1, 11.8, 250.0]
fruits.loc[len(fruits)]=['Currants, black (Ribes nigrum) ', 83.2, 1.5, 0.6, 0.5, 4.0, 9.9, 227.0]
fruits.loc[len(fruits)]=['Custard apple (Annona squamosa) ', 71.5, 1.6, 0.6, 0.6, 5.1, 20.3, 414.0]
fruits.loc[len(fruits)]=['Dates, dry, pale brown (Phoenix dactylifera) ', 11.1, 2.4, 2.2, 0.3, 8.9, 74.9, 1340.0]
fruits.loc[len(fruits)]=['Dates, dry, dark brown (Phoenix dactylifera) ', 13.1, 2.3, 2.3, 0.3, 9.1, 72.6, 1301.0]
fruits.loc[len(fruits)]=['Dates, processed (Phoenix dactylifera) ', 22.0, 1.1, 1.9, 0.4, 6.5, 67.9, 1197.0]
fruits.loc[len(fruits)]=['Fig (Ficus carica) ', 75.6, 2.0, 1.0, 0.3, 4.6, 16.2, 341.0]
fruits.loc[len(fruits)]=['Goosberry (Emblica officinalis) ', 87.0, 0.3, 0.3, 0.1, 7.7, 4.3, 99.0]
fruits.loc[len(fruits)]=['Grapes, seeded, round, black (Vitis vinifera) ', 83.8, 0.7, 0.4, 0.3, 1.3, 13.2, 254.0]
fruits.loc[len(fruits)]=['Grapes, seeded, round, green (Vitis vinifera) ', 85.0, 0.7, 0.4, 0.2, 0.4, 12.1, 235.0]
fruits.loc[len(fruits)]=['Grapes, seeded, round, red (Vitis vinifera) ', 84.4, 0.9, 0.4, 0.2, 1.2, 12.5, 244.0]
fruits.loc[len(fruits)]=['Grapes, seedless, oval, black (Vitis vinifera) ', 75.3, 1.4, 0.7, 0.4, 1.6, 20.4, 395.0]
fruits.loc[len(fruits)]=['Grapes, seedless, round, green (Vitis vinifera) ', 85.5, 0.6, 0.4, 0.2, 1.2, 11.8, 224.0]
fruits.loc[len(fruits)]=['Grapes, seedless, round, black (Vitis vinifera) ', 76.9, 1.2, 0.4, 0.3, 1.1, 19.8, 374.0]
fruits.loc[len(fruits)]=['Guava, white flesh (Psidium guajava) ', 83.7, 1.4, 0.7, 0.3, 8.5, 5.1, 135.0]
fruits.loc[len(fruits)]=['Guava, pink flesh (Psidium guajava) ', 81.2, 1.1, 0.8, 0.2, 7.3, 9.1, 195.0]
fruits.loc[len(fruits)]=['Jack fruit, ripe (Artocarpus heterophyllus) ', 78.5, 2.7, 0.9, 0.1, 3.6, 14.0, 302.0]
fruits.loc[len(fruits)]=['Jambu fruit, ripe (Syzygium cumini) ', 83.3, 0.8, 0.3, 0.1, 3.0, 12.3, 235.0]
fruits.loc[len(fruits)]=['Karonda fruit (Carissa carandas) ', 86.0, 1.1, 1.0, 1.6, 7.2, 2.8, 141.0]
fruits.loc[len(fruits)]=['Lemon, juice (Citrus limon)', 91.59, 0.41, 0.28, 0.75, 6.97,0, 153]
fruits.loc[len(fruits)]=['Lime, sweet,pulp (Citrus limetta) ', 91.3, 0.7, 0.4, 0.2, 2.0, 5.1, 114.0]
fruits.loc[len(fruits)]=['Litchi (Nephelium litchi) ', 85.5, 0.9, 0.4, 0.2, 1.3, 11.4, 225.0]
fruits.loc[len(fruits)]=['Mango, ripe, banganapalli (Mangifera indica) ', 88.4, 0.5, 0.4, 0.5, 1.8, 8.1, 175.0]
fruits.loc[len(fruits)]=['Mango, ripe, gulabkhas (Mangifera indica) ', 86.6, 0.5, 0.3, 0.5, 1.6, 10.3, 209.0]
fruits.loc[len(fruits)]=['Mango, ripe, himsagar (Mangifera indica) ', 88.0, 0.4, 0.3, 0.5, 1.5, 9.0, 187.0]
fruits.loc[len(fruits)]=['Mango, ripe, kesar (Mangifera indica) ', 85.0, 0.5, 0.4, 0.5, 2.0, 11.3, 231.0]
fruits.loc[len(fruits)]=['Mango, ripe, neelam (Mangifera indica) ', 88.4, 0.6, 0.3, 0.5, 1.7, 8.2, 178.0]
fruits.loc[len(fruits)]=['Mango, ripe, paheri (Mangifera indica) ', 87.6, 0.6, 0.4, 0.5, 1.9, 8.6, 188.0]
fruits.loc[len(fruits)]=['Mango, ripe, totapari (Mangifera indica) ', 84.1, 0.4, 0.4, 0.4, 1.7, 12.7, 248.0]
fruits.loc[len(fruits)]=['Mangosteen (Garcinia mangostana) ', 85.5, 0.6, 0.3, 0.2, 1.8, 11.4, 219.0]
fruits.loc[len(fruits)]=['Manila tamarind (Pithecellobium dulce) ', 74.5, 3.5, 2.8, 1.1, 4.4, 13.5, 342.0]
fruits.loc[len(fruits)]=['Musk melon, orange flesh (Cucumis melon) ', 92.9, 0.4, 0.5, 0.3, 1.5, 4.2, 97.0]
fruits.loc[len(fruits)]=['Musk melon, yellow flesh (Cucumis melon) ', 91.8, 0.5, 0.4, 0.2, 1.4, 5.4, 116.0]
fruits.loc[len(fruits)]=['Orange, pulp (Citrus aurantium) ', 89.6, 0.7, 0.3, 0.1, 1.2, 7.9, 156.0]
fruits.loc[len(fruits)]=['Palm fruit, tender (Borassus flabellifer) ', 91.9, 0.5, 0.1, 0.1, 2.4, 4.9, 101.0]
fruits.loc[len(fruits)]=['Papaya, ripe (Carcia papaya) ', 91.4, 0.4, 0.5, 0.1, 2.8, 4.6, 100.0]
fruits.loc[len(fruits)]=['Peach (Prunus communis) ', 88.3, 0.8, 0.5, 0.3, 2.1, 7.8, 168.0]
fruits.loc[len(fruits)]=['Pear (Pyrus sp.) ', 86.4, 0.3, 0.3, 0.2, 4.4, 8.0, 157.0]
fruits.loc[len(fruits)]=['Phalsa (Grewia asiatica) ', 77.4, 1.6, 1.0, 0.1, 4.5, 15.0, 299.0]
fruits.loc[len(fruits)]=['Pineapple (Ananas comosus) ', 86.0, 0.5, 0.3, 0.1, 3.4, 9.4, 180.0]
fruits.loc[len(fruits)]=['Plum (Prunus domestica) ', 84.4, 0.6, 0.3, 0.4, 2.0, 12.1, 238.0]
fruits.loc[len(fruits)]=['Pomegranate, maroon seeds (Punica granatum) ', 83.5, 1.3, 0.5, 0.1, 2.8, 11.5, 229.0]
fruits.loc[len(fruits)]=['Pummelo (Citrus maxima) ', 86.9, 0.6, 0.4, 0.4, 0.8, 10.6, 210.0]
fruits.loc[len(fruits)]=['Raisins, dried, black (Vitis vinifera) ', 19.6, 2.5, 2.1, 0.3, 3.9, 71.2, 1279.0]
fruits.loc[len(fruits)]=['Raisins, dried, golden (Vitis vinifera) ', 21.5, 2.7, 2.0, 0.3, 4.5, 68.7, 1241.0]
fruits.loc[len(fruits)]=['Rambutan (Nephelium lappaceum) ', 80.8, 0.6, 0.4, 0.1, 1.0, 16.8, 306.0]
fruits.loc[len(fruits)]=['Sapota (Achras sapota) ', 73.6, 0.9, 0.6, 1.2, 9.6, 13.9, 307.0]
fruits.loc[len(fruits)]=['Soursop (Annona muricata) ', 80.8, 0.7, 0.5, 0.9, 4.9, 11.9, 260.0]
fruits.loc[len(fruits)]=['Star fruit (Averrhoa carambola) ', 91.1, 0.7, 0.3, 0.3, 2.8, 4.5, 110.0]
fruits.loc[len(fruits)]=['Strawberry (Fragaria ananassa) ', 92.0, 0.9, 0.5, 0.5, 2.5, 3.4, 103.0]
fruits.loc[len(fruits)]=['Tamarind, pulp (Tamarindus indicus) ', 20.8, 2.9, 3.4, 0.1, 5.3, 67.3, 1207.0]
fruits.loc[len(fruits)]=['Water melon, dark green (sugar baby) (Citrullus vulgaris) ', 94.5, 0.6, 0.1, 0.1, 0.7, 3.8, 85.0]
fruits.loc[len(fruits)]=['Water melon, pale green (Citrullus vulgaris) ', 95.3, 0.5, 0.1, 0.1, 0.7, 3.0, 70.0]
fruits.loc[len(fruits)]=['Wood Apple (Limonia acidissima) ', 79.3, 3.1, 1.1, 3.6, 5.2, 7.5, 327.0]
fruits.loc[len(fruits)]=['Zizyphus (Zizyphus jujube) ', 84.3, 1.3, 0.8, 0.3, 3.7, 9.4, 204.0]

print(fruits.head())
roots=pd.DataFrame(columns=dietary_cols)
roots.loc[len(roots)]=['Beet root (Beta vulgaris) ', 86.9, 1.9, 1.4, 0.1, 3.3, 6.1, 149.0]
roots.loc[len(roots)]=['Carrot, orange (Dacus carota) ', 87.6, 0.9, 1.1, 0.4, 4.1, 5.5, 139.0]
roots.loc[len(roots)]=['Carrot, red (Dacus carota) ', 86.0, 1.0, 1.2, 0.4, 4.4, 6.7, 160.0]
roots.loc[len(roots)]=['Colocasia (Colocasia antiquorum) ', 73.4, 3.3, 1.9, 0.1, 3.2, 17.8, 372.0]
roots.loc[len(roots)]=['Lotus root (Nelumbium nelumbo) ', 76.2, 1.9, 1.5, 0.9, 4.7, 14.6, 332.0]
roots.loc[len(roots)]=['Potato, brown skin, big (Solanum tuberosum) ', 80.7, 1.5, 0.9, 0.2, 1.7, 14.8, 292.0]
roots.loc[len(roots)]=['Potato, brown skin, small (Solanum tuberosum) ', 82.9, 1.3, 0.8, 0.2, 1.6, 12.9, 255.0]
roots.loc[len(roots)]=['Potato, red skin (Solanum tuberosum) ', 79.7, 1.8, 1.1, 0.2, 1.6, 15.4, 306.0]
roots.loc[len(roots)]=['Radish, elongate, red skin (Raphanus sativus) ', 89.3, 0.6, 0.7, 0.1, 2.4, 6.7, 134.0]
roots.loc[len(roots)]=['Radish, elongate, white skin (Raphanus sativus) ', 89.0, 0.7, 0.8, 0.1, 2.6, 6.5, 135.0]
roots.loc[len(roots)]=['Radish, round, red skin (Raphanus sativus) ', 89.6, 0.8, 0.9, 0.1, 2.2, 6.0, 130.0]
roots.loc[len(roots)]=['Radish, round, white skin (Raphanus sativus) ', 89.7, 0.8, 0.8, 0.1, 2.3, 6.1, 129.0]
roots.loc[len(roots)]=['Sweet potato, brown skin (Ipomoes batatas) ', 69.2, 1.3, 0.9, 0.2, 3.9, 24.2, 456.0]
roots.loc[len(roots)]=['Sweet potato, pink skin (Ipomoes batatas) ', 69.5, 1.2, 0.9, 0.3, 3.9, 23.9, 452.0]
roots.loc[len(roots)]=['Tapioca (Manihot esculenta) ', 75.2, 1.0, 1.1, 0.2, 4.6, 17.8, 334.0]
roots.loc[len(roots)]=['Water Chestnut (Eleocharis dulcis) ', 73.3, 0.8, 0.9, 0.3, 3.0, 21.4, 400.0]
roots.loc[len(roots)]=['Yam, elephant (Amorphophallus campanulatus) ', 74.3, 2.5, 1.2, 0.1, 4.1, 17.4, 353.0]
roots.loc[len(roots)]=['Yam, ordinary (Amorphophallus campanulatus) ', 74.2, 2.1, 1.6, 0.1, 4.0, 17.6, 349.0]
roots.loc[len(roots)]=['Yam, wild (Dioscorea versicolor) ', 69.3, 3.0, 1.7, 0.3, 4.5, 20.9, 430.0]

print(roots.head())
fresh_spices=pd.DataFrame(columns=dietary_cols)
fresh_spices.loc[len(fresh_spices)]=['Chillies, green-1 (Capsicum annum) ', 84.4, 2.6, 1.0, 0.7, 4.8, 6.3, 191.0]
fresh_spices.loc[len(fresh_spices)]=['Chillies, green-2 (Capsicum annum) ', 85.7, 2.2, 0.8, 0.7, 5.1, 5.2, 167.0]
fresh_spices.loc[len(fresh_spices)]=['Chillies, green-3 (Capsicum annum) ', 85.5, 2.3, 0.9, 0.7, 5.1, 5.2, 169.0]
fresh_spices.loc[len(fresh_spices)]=['Chillies, green-4 (Capsicum annum) ', 85.2, 2.3, 0.8, 0.7, 4.1, 6.6, 190.0]
fresh_spices.loc[len(fresh_spices)]=['Chillies, green-5 (Capsicum annum) ', 85.9, 2.0, 0.8, 0.6, 3.9, 6.6, 180.0]
fresh_spices.loc[len(fresh_spices)]=['Chillies, green-6 (Capsicum annum) ', 84.9, 3.0, 0.9, 0.6, 5.1, 5.3, 175.0]
fresh_spices.loc[len(fresh_spices)]=['Chillies, green-7 (Capsicum annum) ', 85.8, 2.1, 0.9, 0.6, 4.9, 5.6, 163.0]
fresh_spices.loc[len(fresh_spices)]=['Chillies, green - all varieties (Capsicum annum) ', 85.3, 2.3, 0.9, 0.7, 4.7, 5.8, 177.0]
fresh_spices.loc[len(fresh_spices)]=['Coriander leaves (Coriandrum sativum) ', 86.9, 3.5, 2.1, 0.7, 4.6, 1.9, 130.0]
fresh_spices.loc[len(fresh_spices)]=['Curry leaves (Murraya koenigii) ', 65.3, 7.4, 4.8, 1.0, 16.8, 4.5, 266.0]
fresh_spices.loc[len(fresh_spices)]=['Garlic, big clove (Allium sativum) ', 64.3, 6.9, 1.4, 0.1, 5.2, 21.9, 518.0]
fresh_spices.loc[len(fresh_spices)]=['Garlic, small clove (Allium sativum) ', 64.4, 6.7, 1.3, 0.1, 5.4, 21.8, 514.0]
fresh_spices.loc[len(fresh_spices)]=['Garlic, single clove, Kashmir (Allium sativum) ', 64.4, 6.1, 1.7, 0.1, 4.0, 23.4, 523.0]
fresh_spices.loc[len(fresh_spices)]=['Ginger, fresh (Zinziber officinale) ', 81.2, 2.2, 1.3, 0.8, 5.3, 8.9, 230.0]
fresh_spices.loc[len(fresh_spices)]=['Mango ginger (Curcuma amada) ', 84.5, 1.4, 1.5, 0.7, 4.7, 6.9, 177.0]
fresh_spices.loc[len(fresh_spices)]=['Mint leaves (Mentha spicata ) ', 84.2, 4.6, 2.1, 0.6, 5.8, 2.3, 155.0]
fresh_spices.loc[len(fresh_spices)]=['Onion, big (Allium cepa) ', 85.7, 1.5, 0.5, 0.2, 2.4, 9.5, 201.0]
fresh_spices.loc[len(fresh_spices)]=['Onion, small (Allium cepa) ', 84.6, 1.8, 0.6, 0.1, 1.1, 11.5, 237.0]

print(fresh_spices.head())
dry_spices=pd.DataFrame(columns=dietary_cols)
dry_spices.loc[len(dry_spices)]=['Asafoetida (Ferula assa-foetida) ', 9.4, 6.3, 5.9, 1.2, 5.1, 71.9, 1387.0]
dry_spices.loc[len(dry_spices)]=['Cardamom, green (Elettaria cardamomum) ', 11.2, 8.1, 7.2, 2.6, 23.1, 47.7, 1067.0]
dry_spices.loc[len(dry_spices)]=['Cardamom, black (Elettaria cardamomum) ', 6.6, 6.6, 7.8, 2.8, 23.4, 52.5, 1132.0]
dry_spices.loc[len(dry_spices)]=['Chillies, red (Capsicum annum) ', 14.5, 12.6, 5.7, 6.4, 31.1, 29.4, 990.0]
dry_spices.loc[len(dry_spices)]=['Cloves (Syzygium aromaticum) ', 26.4, 5.8, 5.9, 8.4, 34.5, 18.7, 781.0]
dry_spices.loc[len(dry_spices)]=['Coriander seeds (Coriandrum sativum) ', 8.7, 10.6, 5.3, 17.4, 44.8, 12.9, 1125.0]
dry_spices.loc[len(dry_spices)]=['Cumin seeds (Cuminum cyminum) ', 10.5, 13.9, 5.9, 16.6, 30.3, 22.6, 1274.0]
dry_spices.loc[len(dry_spices)]=['Fenugreek seeds (Trigonella foenum graecum) ', 7.8, 25.4, 2.9, 5.7, 47.5, 10.5, 983.0]
dry_spices.loc[len(dry_spices)]=['Mace (Myristica fragrans) ', 20.0, 6.2, 2.4, 24.4, 20.3, 26.5, 1488.0]
dry_spices.loc[len(dry_spices)]=['Nutmeg (Myristica fragrans) ', 15.5, 6.3, 1.9, 36.5, 11.9, 27.6, 1940.0]
dry_spices.loc[len(dry_spices)]=['Omum (Trachyspermum ammi) ', 9.7, 15.8, 8.1, 21.1, 20.5, 24.5, 1495.0]
dry_spices.loc[len(dry_spices)]=['Pippali (Piper longum) ', 10.9, 10.5, 6.4, 2.2, 34.1, 35.7, 906.0]
dry_spices.loc[len(dry_spices)]=['Pepper, black (Piper nigrum) ', 13.1, 10.1, 4.5, 2.7, 33.1, 36.2, 910.0]
dry_spices.loc[len(dry_spices)]=['Poppy seeds (Papaver somniferum) ', 4.2, 20.3, 6.0, 30.3, 26.6, 12.3, 1768.0]
dry_spices.loc[len(dry_spices)]=['Turmeric powder (Curcuma domestica) ', 10.5, 7.6, 6.1, 5.0, 21.3, 49.2, 1174.0]

print(dry_spices.head())
nuts=pd.DataFrame(columns=dietary_cols)
nuts.loc[len(nuts)]=['Almond (Prunus amygdalus) ', 4.3, 18.4, 2.6, 58.4, 13.0, 3.0, 2549.0]
nuts.loc[len(nuts)]=['Arecanut, dried, brown (Areca catechu) ', 6.6, 5.7, 1.3, 4.3, 11.4, 70.4, 1467.0]
nuts.loc[len(nuts)]=['Arecanut, dried, red color (Areca catechu) ', 6.3, 6.4, 1.4, 4.4, 11.1, 70.2, 1477.0]
nuts.loc[len(nuts)]=['Arecanut, fresh (Areca catechu) ', 37.7, 2.7, 1.4, 5.5, 7.6, 45.0, 1024.0]
nuts.loc[len(nuts)]=['Cashew nut (Anacardium occidentale) ', 4.4, 18.7, 2.2, 45.2, 3.8, 25.4, 2438.0]
nuts.loc[len(nuts)]=['Coconut, kernal, dry (Cocos nucifera) ', 3.9, 7.2, 1.6, 63.2, 15.8, 8.0, 2611.0]
nuts.loc[len(nuts)]=['Coconut, kernel, fresh (Cocos nucifera) ', 36.1, 3.8, 1.9, 41.3, 10.4, 6.3, 1711.0]
nuts.loc[len(nuts)]=['Garden cress, seeds (Lepidium sativum) ', 4.6, 23.3, 6.3, 23.7, 8.2, 33.6, 1863.0]
nuts.loc[len(nuts)]=['Gingelly seeds, black (Sesamum indicum) ', 4.5, 19.1, 5.7, 43.1, 17.1, 10.2, 2124.0]
nuts.loc[len(nuts)]=['Gingelly seeds, brown (Sesamum indicum) ', 3.6, 21.6, 4.5, 43.2, 17.2, 9.7, 2161.0]
nuts.loc[len(nuts)]=['Gingelly seeds, white (Sesamum indicum) ', 3.3, 21.7, 4.1, 43.0, 16.9, 10.8, 2174.0]
nuts.loc[len(nuts)]=['Ground nut (Arachis hypogea) ', 6.9, 23.6, 2.1, 39.6, 10.3, 17.2, 2176.0]
nuts.loc[len(nuts)]=['Mustard seeds (Brassica juncea) ', 5.6, 19.5, 3.7, 40.1, 14.1, 16.8, 2132.0]
nuts.loc[len(nuts)]=['Linseeds (Linum usitatissimum) ', 5.4, 18.5, 3.1, 35.6, 26.1, 10.9, 1857.0]
nuts.loc[len(nuts)]=['Niger seeds, black (Guizotia abyssinica) ', 4.6, 18.9, 3.9, 38.6, 10.9, 22.9, 2144.0]
nuts.loc[len(nuts)]=['Niger seeds, gray (Guizotia abyssinica) ', 5.6, 18.3, 4.9, 39.5, 10.9, 20.5, 2128.0]
nuts.loc[len(nuts)]=['Pine seed (Pinus gerardiana) ', 5.3, 12.5, 2.7, 48.7, 3.7, 26.7, 2486.0]
nuts.loc[len(nuts)]=['Pistachio nuts (Pistacla vera) ', 4.6, 23.3, 3.0, 42.4, 10.6, 15.8, 2257.0]
nuts.loc[len(nuts)]=['Safflower seeds (Carthamus tinctorius) ', 5.2, 17.6, 2.5, 30.8, 13.4, 30.1, 1981.0]
nuts.loc[len(nuts)]=['Sunflower seeds (Helianthus annuus) ', 3.5, 23.5, 3.4, 51.8, 10.8, 6.8, 2453.0]
nuts.loc[len(nuts)]=['Walnut (Juglans regia) ', 3.5, 14.9, 1.7, 64.2, 5.3, 10.1, 2809.0]

print(nuts.head())
sugars=pd.DataFrame(columns=dietary_cols)
sugars.loc[len(sugars)]=['Jaggery, cane (Saccharum officinarum) ', 11.2, 1.8, 1.9, 0.1, 0.0, 84.8, 1480.0]
sugars.loc[len(sugars)]=['Sugarcane, juice (Saccharum officinarum) ', 85.5, 0.1, 0.2, 0.4, 0.5, 13.1, 242.0]

print(sugars.head())
mushroom=pd.DataFrame(columns=dietary_cols)
mushroom.loc[len(mushroom)]=['Button mushroom, fresh (Agaricus sp.) ', 90.0, 3.6, 0.7, 0.4, 3.1, 1.9, 115.0]
mushroom.loc[len(mushroom)]=['Chicken mushroom, fresh (Lactiporus sp.) ', 92.4, 1.8, 0.7, 0.2, 1.9, 2.7, 89.0]
mushroom.loc[len(mushroom)]=['Shiitake mushroom, fresh (Lentinula sp.) ', 82.9, 3.1, 1.1, 0.7, 3.0, 8.9, 243.0]
mushroom.loc[len(mushroom)]=['Oyster mushroom, dried (Pleurotus sp.) ', 4.5, 19.0, 1.4, 2.8, 39.1, 33.0, 1019.0]

print(mushroom.head())
misc=pd.DataFrame(columns=dietary_cols)
misc.loc[len(misc)]=['Toddy ', 93.8, 0.1, 0.2, 0.0, 0.0, 5.7, 101.0]
misc.loc[len(misc)]=['Coconut Water ', 95.7, 0.2, 0.6, 0.1, 0.0, 3.1, 64.0]

print(misc.head())
milk=pd.DataFrame(columns=dietary_cols)
milk.loc[len(milk)]=['Milk, whole, Buffalo ', 80.6, 3.6, 0.6, 6.5, 0, 8.3, 449.0]
milk.loc[len(milk)]=['Milk, whole, Cow ', 86.6, 3.2, 0.6, 4.4, 0, 4.9, 305.0]
milk.loc[len(milk)]=['Panner ', 51.9, 18.8, 1.9, 14.7, 0, 12.4, 1079.0]
milk.loc[len(milk)]=['Khoa ', 42.5, 16.3, 4.0, 20.6, 0, 16.5, 1322.0]

print(milk.head())
egg=pd.DataFrame(columns=dietary_cols)
egg.loc[len(egg)]=['Egg, poultry, whole, raw ', 76.5, 13.2, 0.8, 9.1,0,0, 564.0]
egg.loc[len(egg)]=['Egg, poultry, white, raw ', 86.6, 10.8, 0.7, 0.0,0,0, 187.0]
egg.loc[len(egg)]=['Egg, poultry, yolk, raw ', 53.5, 15.7, 1.0, 26.3,0,0, 1242.0]
egg.loc[len(egg)]=['Egg, poultry, whole, boiled ', 73.4, 13.4, 0.8, 10.5,0,0, 618.0]
egg.loc[len(egg)]=['Egg, poultry, white, boiled ', 83.5, 12.3, 0.8, 0.20,0,0, 220.0]
egg.loc[len(egg)]=['Egg, poultry, yolk, boiled ', 51.4, 16.1, 1.3,  27.4,0,0, 1290.0]
egg.loc[len(egg)]=['Egg, poultry, omlet ', 68.4, 16.5, 0.9,  11.6,0,0, 710.0]
egg.loc[len(egg)]=['Egg,country hen, whole, raw ', 72.9, 13.1, 0.8,  13.0, 0,0,704.0]
egg.loc[len(egg)]=['Egg,country hen, whole, boiled ', 70.4, 14.4, 0.9, 14.1,0,0, 767.0]
egg.loc[len(egg)]=['Egg,country hen, omlet ', 67.8, 14.8, 1.0,  16.3,0,0, 855.0]
egg.loc[len(egg)]=['Egg, duck,whole, boiled ', 71.6, 13.8, 0.9, 13.6,0,0, 738.0]
egg.loc[len(egg)]=['Egg, duck,whole, raw ', 70.5, 14.6, 0.9, 13.8,0,0, 760.0]
egg.loc[len(egg)]=['Egg, duck,whole, omlet ', 68.9, 15.1, 1.1, 14.8,0,0, 804.0]
egg.loc[len(egg)]=['Egg, quial, whole, raw ', 75.1, 12.3, 0.9, 11.4,0,0, 635.0]
egg.loc[len(egg)]=['Egg, quial, whole, boiled ', 74.4, 13.0, 0.9, 11.5,0,0, 647.0]

print(egg.head())
poultry=pd.DataFrame(columns=dietary_cols)
poultry.loc[len(poultry)]=['Chicken, poultry, leg,skinless ', 67.6, 19.4, 1.1, 12.6, 0, 0, 1605.0]
poultry.loc[len(poultry)]=['Chicken, poultry,thigh,skinless ', 67.5, 18.1, 1.1, 14.2, 0, 0, 836.0]
poultry.loc[len(poultry)]=['Chicken, poultry, breast, skinless ', 67.1, 21.8, 1.1, 9.0, 0, 0, 704.0]
poultry.loc[len(poultry)]=['Chicken, poultry,wing, skinless ', 67.4, 17.4, 1.1, 13.8, 0, 0, 807.0]
poultry.loc[len(poultry)]=['Poultry, chicken, liver ', 73.2, 21.5, 1.1, 4.0, 0, 0, 518.0]
poultry.loc[len(poultry)]=['Poultry, chicken, gizzard ', 78.0, 18.2, 1.4, 2.0, 0, 0, 386.0]
poultry.loc[len(poultry)]=['Country hen, leg,withskin ', 70.0, 17.0, 1.2, 11.7, 0, 0, 723.0]
poultry.loc[len(poultry)]=['Country hen,thigh,withskin ', 67.4, 18.2, 1.2, 12.8, 0, 0, 785.0]
poultry.loc[len(poultry)]=['Country hen, breast,withskin ', 66.5, 22.0, 1.1, 10.2, 0, 0, 753.0]
poultry.loc[len(poultry)]=['Country hen,wing, with skin ', 68.0, 18.6, 1.1, 12.0, 0, 0, 764.0]
poultry.loc[len(poultry)]=['Duck, meat, withskin ', 73.4, 19.0, 1.2, 6.0, 0, 0, 547.0]
poultry.loc[len(poultry)]=['Emu, meat, skinless ', 71.6, 22.6, 0.9, 4.6, 0, 0, 556.0]
poultry.loc[len(poultry)]=['Guinea fowl, meat,withskin ', 75.2, 20.5, 0.9, 3.2, 0, 0, 469.0]
poultry.loc[len(poultry)]=['Pigeon, meat, with skin ', 74.6, 17.9, 1.3, 6.0, 0, 0, 528.0]
poultry.loc[len(poultry)]=['Quail, meat,skinless ', 71.5, 20.9, 1.3, 5.9, 0, 0, 576.0]
poultry.loc[len(poultry)]=['Turkey, leg,withskin ', 69.2, 20.3, 1.2, 8.1, 0, 0, 647.0]
poultry.loc[len(poultry)]=['Turkey,thigh,withskin ', 72.1, 20.4, 0.9, 6.3, 0, 0, 581.0]
poultry.loc[len(poultry)]=['Turkey, breast, withskin ', 68.4, 21.9, 1.0, 8.0, 0, 0, 671.0]
poultry.loc[len(poultry)]=['Turkey,wing,with skin ', 66.1, 21.9, 0.9, 10.7, 0, 0, 771.0]

print(poultry.head())
animal=pd.DataFrame(columns=dietary_cols)
animal.loc[len(animal)]=['Goat, shoulder ', 66.3, 20.3, 0.9, 11.9, 0, 0, 787.0]
animal.loc[len(animal)]=['Goat, chops ', 72.4, 20.3, 0.9, 5.9, 0, 0, 568.0]
animal.loc[len(animal)]=['Goat, legs ', 68.8, 22.0, 0.9, 7.9, 0, 0, 669.0]
animal.loc[len(animal)]=['Goat, brain ', 76.7, 13.8, 1.2, 8.0, 0, 0, 533.0]
animal.loc[len(animal)]=['Goat, tongue ', 68.4, 16.6, 1.0, 13.6, 0, 0, 789.0]
animal.loc[len(animal)]=['Goat, lungs ', 79.0, 16.8, 0.7, 3.0, 0, 0, 401.0]
animal.loc[len(animal)]=['Goat, heart ', 75.1, 19.3, 0.9, 4.4, 0, 0, 492.0]
animal.loc[len(animal)]=['Goat, liver ', 73.3, 20.3, 1.3, 4.8, 0, 0, 526.0]
animal.loc[len(animal)]=['Goat, tripe ', 80.9, 15.3, 0.2, 3.3, 0, 0, 386.0]
animal.loc[len(animal)]=['Goat, spleen ', 77.9, 18.4, 1.1, 2.3, 0, 0, 401.0]
animal.loc[len(animal)]=['Goat, kidneys ', 80.2, 15.6, 1.1, 2.9, 0, 0, 374.0]
animal.loc[len(animal)]=['Goat, tube (small intestine) ', 78.1, 12.9, 0.5, 8.2, 0, 0, 525.0]
animal.loc[len(animal)]=['Goat, testis ', 84.2, 12.3, 0.8, 2.3, 0, 0, 298.0]
animal.loc[len(animal)]=['Sheep,shoulder ', 66.5, 18.2, 0.8, 14.3, 0, 0, 840.0]
animal.loc[len(animal)]=['Sheep,chops ', 75.6, 18.0, 1.0, 5.1, 0, 0, 496.0]
animal.loc[len(animal)]=['Sheep, leg ', 68.2, 21.4, 0.9, 8.6, 0, 0, 686.0]
animal.loc[len(animal)]=['Sheep, brain ', 78.3, 13.0, 1.1, 7.2, 0, 0, 492.0]
animal.loc[len(animal)]=['Sheep,tongue ', 68.7, 16.6, 1.0, 13.5, 0, 0, 783.0]
animal.loc[len(animal)]=['Sheep, lungs ', 80.4, 16.1, 0.9, 2.4, 0, 0, 363.0]
animal.loc[len(animal)]=['Sheep, heart ', 77.0, 18.1, 0.9, 3.6, 0, 0, 445.0]
animal.loc[len(animal)]=['Sheep, liver ', 69.7, 22.2, 1.2, 4.8, 0, 0, 559.0]
animal.loc[len(animal)]=['Sheep,tripe ', 78.8, 16.7, 0.2, 4.0, 0, 0, 435.0]
animal.loc[len(animal)]=['Sheep,spleen ', 79.6, 16.0, 1.1, 3.0, 0, 0, 384.0]
animal.loc[len(animal)]=['Sheep,kidneys ', 79.7, 16.2, 1.0, 2.9, 0, 0, 384.0]
animal.loc[len(animal)]=['Beef, shoulder ', 63.8, 20.5, 0.9, 14.5, 0, 0, 889.0]
animal.loc[len(animal)]=['Beef, chops ', 72.4, 19.8, 0.9, 6.7, 0, 0, 585.0]
animal.loc[len(animal)]=['Beef, round(leg) ', 68.0, 22.6, 1.1, 7.3, 0, 0, 658.0]
animal.loc[len(animal)]=['Beef, brain ', 78.7, 10.5, 1.3, 9.2, 0, 0, 523.0]
animal.loc[len(animal)]=['Beef, tongue ', 70.9, 15.6, 0.7, 12.5, 0, 0, 731.0]
animal.loc[len(animal)]=['Beef, lungs ', 80.8, 15.6, 1.1, 2.2, 0, 0, 351.0]
animal.loc[len(animal)]=['Beef, heart ', 77.7, 17.6, 0.8, 3.5, 0, 0, 433.0]
animal.loc[len(animal)]=['Beef, liver ', 74.1, 20.7, 1.0, 3.9, 0, 0, 499.0]
animal.loc[len(animal)]=['Beef, tripe ', 83.7, 13.1, 0.5, 2.5, 0, 0, 316.0]
animal.loc[len(animal)]=['Beef, spleen ', 79.0, 17.4, 1.2, 2.2, 0, 0, 378.0]
animal.loc[len(animal)]=['Beef, kidneys ', 77.5, 17.0, 1.2, 4.0, 0, 0, 439.0]
animal.loc[len(animal)]=['Calf, shoulder ', 70.4, 20.9, 0.9, 7.4, 0, 0, 633.0]
animal.loc[len(animal)]=['Calf, chops ', 72.6, 22.4, 0.9, 3.8, 0, 0, 524.0]
animal.loc[len(animal)]=['Calf, round(leg) ', 71.1, 21.1, 0.7, 6.9, 0, 0, 615.0]
animal.loc[len(animal)]=['Calf, brain ', 81.1, 9.8, 1.2, 7.6, 0, 0, 448.0]
animal.loc[len(animal)]=['Calf, tongue ', 69.4, 17.7, 0.9, 11.7, 0, 0, 737.0]
animal.loc[len(animal)]=['Calf, heart ', 73.4, 18.8, 0.8, 3.7, 0, 0, 459.0]
animal.loc[len(animal)]=['Calf, liver ', 73.6, 21.0, 1.2, 3.9, 0, 0, 503.0]
animal.loc[len(animal)]=['Calf, spleen ', 78.6, 17.7, 1.4, 2.1, 0, 0, 379.0]
animal.loc[len(animal)]=['Calf, kidneys ', 80.2, 15.1, 1.0, 3.5, 0, 0, 387.0]
animal.loc[len(animal)]=['Mithun, shoulder ', 68.8, 19.0, 0.8, 11.1, 0, 0, 736.0]
animal.loc[len(animal)]=['Mithun, chops ', 73.8, 18.1, 0.7, 6.1, 0, 0, 536.0]
animal.loc[len(animal)]=['Mithun, round (leg) ', 72.2, 19.6, 0.9, 4.1, 0, 0, 485.0]
animal.loc[len(animal)]=['Pork, shoulder ', 62.9, 17.4, 0.7, 18.8, 0, 0, 993.0]
animal.loc[len(animal)]=['Pork, chops ', 68.4, 19.4, 0.6, 11.3, 0, 0, 748.0]
animal.loc[len(animal)]=['Pork, ham ', 61.9, 18.8, 0.6, 18.5, 0, 0, 1006.0]
animal.loc[len(animal)]=['Pork, lungs ', 81.1, 15.1, 0.9, 2.7, 0, 0, 358.0]
animal.loc[len(animal)]=['Pork, heart ', 77.7, 16.3, 1.0, 4.8, 0, 0, 457.0]
animal.loc[len(animal)]=['Pork, liver ', 74.8, 19.8, 1.1, 3.9, 0, 0, 484.0]
animal.loc[len(animal)]=['Pork, stomach ', 76.5, 15.3, 0.2, 7.8, 0, 0, 550.0]
animal.loc[len(animal)]=['Pork, spleen ', 80.9, 15.1, 1.1, 2.7, 0, 0, 357.0]
animal.loc[len(animal)]=['Pork, kidneys ', 80.9, 14.3, 0.9, 3.6, 0, 0, 379.0]
animal.loc[len(animal)]=['Pork,tube(small intestine) ', 75.4, 14.9, 0.5, 8.9, 0, 0, 587.0]
animal.loc[len(animal)]=['Hare,shoulder ', 71.0, 21.1, 1.1, 6.5, 0, 0, 603.0]
animal.loc[len(animal)]=['Hare,chops ', 75.4, 20.6, 1.2, 2.5, 0, 0, 445.0]
animal.loc[len(animal)]=['Hare, leg ', 73.0, 20.5, 1.2, 4.1, 0, 0, 503.0]
animal.loc[len(animal)]=['Rabbit, shoulder ', 70.7, 20.0, 1.1, 7.9, 0, 0, 635.0]
animal.loc[len(animal)]=['Rabbit, chops ', 71.0, 22.6, 1.3, 4.8, 0, 0, 565.0]
animal.loc[len(animal)]=['Rabbit, leg ', 70.3, 21.3, 1.2, 5.9, 0, 0, 584.0]

print(animal.head())
marine_fish=pd.DataFrame(columns=dietary_cols)
marine_fish.loc[len(marine_fish)]=['Allathi (Elops machnata) ', 75.9, 21.7, 1.0, 0.9, 0, 0, 406.0]
marine_fish.loc[len(marine_fish)]=['Aluva (Parastromateus niger) ', 75.7, 21.6, 1.3, 1.8, 0, 0, 434.0]
marine_fish.loc[len(marine_fish)]=['Anchovy (Stolephorus indicus) ', 77.7, 19.8, 1.6, 0.7, 0, 0, 367.0]
marine_fish.loc[len(marine_fish)]=['Arifish (Aprion virescens) ', 77.0, 22.0, 1.3, 1.1, 0, 0, 415.0]
marine_fish.loc[len(marine_fish)]=['Betki (Lates calcarifer) ', 82.5, 15.2, 1.1, 0.2, 0, 0, 284.0]
marine_fish.loc[len(marine_fish)]=['Black snapper (Macolorniger) ', 78.0, 19.5, 1.1, 1.2, 0, 0, 377.0]
marine_fish.loc[len(marine_fish)]=['Bombay duck (Harpadon nehereus) ', 83.2, 13.5, 1.0, 1.0, 0, 0, 287.0]
marine_fish.loc[len(marine_fish)]=['Bommuralu (Muraenesox cinerius) ', 76.4, 22.3, 1.2, 2.8, 0, 0, 485.0]
marine_fish.loc[len(marine_fish)]=['Cat fish (Tachysurus thalassinus) ', 76.2, 22.1, 1.3, 2.1, 0, 0, 456.0]
marine_fish.loc[len(marine_fish)]=['Chakla (Rachycentron canadum) ', 78.3, 20.2, 1.0, 1.6, 0, 0, 406.0]
marine_fish.loc[len(marine_fish)]=['Chappal (Aluterus monoceros) ', 80.6, 17.1, 1.2, 0.6, 0, 0, 317.0]
marine_fish.loc[len(marine_fish)]=['Chelu (Elagatisbipinnulata) ', 76.3, 20.0, 1.2, 0.7, 0, 0, 366.0]
marine_fish.loc[len(marine_fish)]=['Chembali (Lutjanus quinquelineatus) ', 77.5, 20.0, 0.8, 1.7, 0, 0, 406.0]
marine_fish.loc[len(marine_fish)]=['Eri meen (Pristipomoides filamentosus) ', 76.5, 22.3, 1.2, 2.0, 0, 0, 454.0]
marine_fish.loc[len(marine_fish)]=['Gobro (Epinephelusdiacanthus) ', 78.8, 19.3, 0.9, 0.8, 0, 0, 362.0]
marine_fish.loc[len(marine_fish)]=['Guitar fish (Rhinobatusprahli) ', 75.7, 22.5, 1.3, 0.4, 0, 0, 400.0]
marine_fish.loc[len(marine_fish)]=['Hilsa (Tenualosa ilisha) ', 60.0, 21.8, 1.1, 18.4, 0, 0, 1083.0]
marine_fish.loc[len(marine_fish)]=['Jallal (Arius sp.) ', 77.5, 21.5, 1.1, 1.4, 0, 0, 420.0]
marine_fish.loc[len(marine_fish)]=['Jathivela meen (Lethrinus lentjan) ', 75.6, 22.4, 1.2, 1.9, 0, 0, 453.0]
marine_fish.loc[len(marine_fish)]=['Kadal bral (Synodus indicus) ', 79.6, 18.7, 1.2, 1.3, 0, 0, 368.0]
marine_fish.loc[len(marine_fish)]=['Kadali (Nemipterus mesoprion) ', 73.7, 22.0, 1.0, 4.2, 0, 0, 530.0]
marine_fish.loc[len(marine_fish)]=['Kalamaara (Leptomelanosoma indicum) ', 75.5, 20.8, 1.0, 4.5, 0, 0, 523.0]
marine_fish.loc[len(marine_fish)]=['Kalava (Epinephelus coioides) ', 80.0, 19.3, 1.3, 1.2, 0, 0, 374.0]
marine_fish.loc[len(marine_fish)]=['Kanamayya (Lutjanus rivulatus) ', 77.5, 20.1, 1.0, 0.6, 0, 0, 367.0]
marine_fish.loc[len(marine_fish)]=['Kannadi paarai (Alectis indicus) ', 76.2, 22.0, 1.2, 1.2, 0, 0, 423.0]
marine_fish.loc[len(marine_fish)]=['Karimeen (Etroplus suratensis) ', 78.6, 19.6, 0.9, 1.3, 0, 0, 386.0]
marine_fish.loc[len(marine_fish)]=['Karnagawala (Anchoa hepsetus) ', 79.2, 19.5, 1.4, 0.8, 0, 0, 363.0]
marine_fish.loc[len(marine_fish)]=['Kayrai (Thunnusalbacores) ', 72.6, 20.1, 1.2, 3.0, 0, 0, 454.0]
marine_fish.loc[len(marine_fish)]=['Kiriyan (Atule mate) ', 72.3, 22.4, 0.9, 4.7, 0, 0, 556.0]
marine_fish.loc[len(marine_fish)]=['Kitefish (Mobula kuhlii) ', 77.6, 23.0, 1.5, 0.5, 0, 0, 413.0]
marine_fish.loc[len(marine_fish)]=['Korka (Terapon jarbua) ', 72.2, 23.7, 1.0, 3.3, 0, 0, 528.0]
marine_fish.loc[len(marine_fish)]=['Kulam paarai (Carangoides fulvoguttatus) ', 73.4, 21.6, 1.2, 3.6, 0, 0, 501.0]
marine_fish.loc[len(marine_fish)]=['Maagaa (Polynemus plebeius) ', 79.2, 20.1, 1.0, 0.6, 0, 0, 366.0]
marine_fish.loc[len(marine_fish)]=['Mackerel (Rastrelliger kanagurta) ', 74.5, 21.5, 1.2, 1.5, 0, 0, 423.0]
marine_fish.loc[len(marine_fish)]=['Mandaclathi (Nasoreticulatus) ', 76.3, 21.2, 1.4, 1.0, 0, 0, 398.0]
marine_fish.loc[len(marine_fish)]=['Matha (Acanthurus mata) ', 79.8, 21.1, 1.3, 0.7, 0, 0, 389.0]
marine_fish.loc[len(marine_fish)]=['Milkfish (Chanos chanos) ', 72.2, 23.6, 1.1, 1.0, 0, 0, 442.0]
marine_fish.loc[len(marine_fish)]=['Moon fish (Menemaculata) ', 74.2, 20.7, 1.6, 4.6, 0, 0, 526.0]
marine_fish.loc[len(marine_fish)]=['Mullet (Mugil cephalus) ', 76.1, 20.2, 1.1, 1.3, 0, 0, 393.0]
marine_fish.loc[len(marine_fish)]=['Mural (Tylosuruscrocodilus crocodilus) ', 78.9, 19.0, 1.1, 0.5, 0, 0, 345.0]
marine_fish.loc[len(marine_fish)]=['Myil meen (Istiophorusplatypterus) ', 75.5, 22.8, 1.2, 0.5, 0, 0, 410.0]
marine_fish.loc[len(marine_fish)]=['Nalla bontha (Epinephelussp.) ', 79.3, 19.8, 1.0, 0.7, 0, 0, 364.0]
marine_fish.loc[len(marine_fish)]=['Narba (Caranx sexfasciatus) ', 76.3, 21.9, 1.1, 1.6, 0, 0, 432.0]
marine_fish.loc[len(marine_fish)]=['Paarai (Caranxheberi) ', 75.7, 21.5, 1.2, 1.8, 0, 0, 435.0]
marine_fish.loc[len(marine_fish)]=['Padayappa (Canthidermis maculata) ', 79.1, 19.7, 1.2, 0.6, 0, 0, 360.0]
marine_fish.loc[len(marine_fish)]=['Pali kora (Pannamicrodon) ', 78.6, 19.2, 1.0, 1.8, 0, 0, 394.0]
marine_fish.loc[len(marine_fish)]=['Pambada (Lepturacanthus savala) ', 74.6, 21.9, 1.4, 4.4, 0, 0, 537.0]
marine_fish.loc[len(marine_fish)]=['Pandukopa (Pseudosciaena manchurica) ', 77.4, 19.7, 1.1, 0.6, 0, 0, 360.0]
marine_fish.loc[len(marine_fish)]=['Parava (Lactarius lactarius) ', 77.1, 21.5, 1.0, 2.7, 0, 0, 467.0]
marine_fish.loc[len(marine_fish)]=['Parcus (Psettodes erumei) ', 79.0, 19.8, 1.1, 0.6, 0, 0, 361.0]
marine_fish.loc[len(marine_fish)]=['Parrotfish (Scarus ghobban) ', 76.8, 20.8, 1.4, 0.5, 0, 0, 375.0]
marine_fish.loc[len(marine_fish)]=['Perinkilichai (Pinjalo pinjalo) ', 78.3, 20.8, 1.0, 0.8, 0, 0, 387.0]
marine_fish.loc[len(marine_fish)]=['Phopat (Coryphaenahippurus) ', 76.4, 22.0, 1.2, 1.3, 0, 0, 424.0]
marine_fish.loc[len(marine_fish)]=['Piranha (Pygopritissp.) ', 76.1, 20.4, 0.9, 0, 0, 5.4, 549.0]
marine_fish.loc[len(marine_fish)]=['Pomfret, black (Parastromateus niger) ', 74.4, 18.9, 0.9, 4.8, 0, 0, 515.0]
marine_fish.loc[len(marine_fish)]=['Pomfret, snub nose (Trachinotus blochii) ', 77.8, 21.0, 1.2, 0.4, 0, 0, 375.0]
marine_fish.loc[len(marine_fish)]=['Pomfret, white (Pampus argenteus) ', 75.9, 19.0, 1.0, 5.1, 0, 0, 513.0]
marine_fish.loc[len(marine_fish)]=['Pranel (Gerres sp.) ', 79.5, 19.6, 0.7, 1.8, 0, 0, 405.0]
marine_fish.loc[len(marine_fish)]=['Pulli paarai (Gnathanodon speciosus) ', 75.9, 20.0, 1.1, 1.5, 0, 0, 399.0]
marine_fish.loc[len(marine_fish)]=['Queenfish (Scomberoides commersonianus) ', 76.6, 20.9, 1.2, 1.1, 0, 0, 400.0]
marine_fish.loc[len(marine_fish)]=['Raaifish (Lobotes surinamensis) ', 77.1, 21.6, 1.1, 1.6, 0, 0, 430.0]
marine_fish.loc[len(marine_fish)]=['Raaivanthu (Epinephelus chlorostigma) ', 79.9, 19.3, 0.9, 2.1, 0, 0, 410.0]
marine_fish.loc[len(marine_fish)]=['Rani (Pinkperch) ', 78.5, 18.8, 1.0, 1.4, 0, 0, 377.0]
marine_fish.loc[len(marine_fish)]=['Rayfish, bow head,spotted (Rhina ancylostoma) ', 80.3, 19.0, 1.1, 0.7, 0, 0, 349.0]
marine_fish.loc[len(marine_fish)]=['Redsnapper (Lutjanus argentimaculatus) ', 76.2, 22.7, 1.2, 1.3, 0, 0, 437.0]
marine_fish.loc[len(marine_fish)]=['Redsnapper,small (Priacanthus hamrur) ', 76.3, 21.5, 1.1, 2.3, 0, 0, 451.0]
marine_fish.loc[len(marine_fish)]=['Sadaya (Platax orbicularis) ', 75.5, 20.6, 1.3, 2.9, 0, 0, 462.0]
marine_fish.loc[len(marine_fish)]=['Salmon (Salmo salar) ', 67.8, 20.9, 1.1, 9.8, 0, 0, 721.0]
marine_fish.loc[len(marine_fish)]=['Sangada (Nemipterus japanicus) ', 78.5, 20.2, 1.2, 2.6, 0, 0, 443.0]
marine_fish.loc[len(marine_fish)]=['Sankata paarai (Caranx ignobilis) ', 74.8, 21.8, 1.1, 1.6, 0, 0, 434.0]
marine_fish.loc[len(marine_fish)]=['Sardine (Sardinella longiceps) ', 72.2, 17.9, 0.8, 8.9, 0, 0, 637.0]
marine_fish.loc[len(marine_fish)]=['Shark (Carcharhinus sorrah) ', 72.8, 21.6, 1.0, 0.8, 0, 0, 398.0]
marine_fish.loc[len(marine_fish)]=['Shark, hammer head (Sphyrnamokarran) ', 74.6, 23.4, 0.9, 0.8, 0, 0, 432.0]
marine_fish.loc[len(marine_fish)]=['Shark,spotted (Stegostoma fasciatum) ', 78.8, 20.9, 1.0, 0.7, 0, 0, 384.0]
marine_fish.loc[len(marine_fish)]=['Shelavu (Sphyraena jello) ', 74.6, 22.4, 1.2, 1.7, 0, 0, 446.0]
marine_fish.loc[len(marine_fish)]=['Silan (Silonia silondia) ', 70.1, 22.7, 0.7, 6.6, 0, 0, 633.0]
marine_fish.loc[len(marine_fish)]=['Silk fish (Beryxsp.) ', 77.6, 20.6, 1.0, 1.5, 0, 0, 408.0]
marine_fish.loc[len(marine_fish)]=['Silver carp (Hypophthalmichthysmolitrix) ', 72.4, 21.7, 0.9, 5.1, 0, 0, 555.0]
marine_fish.loc[len(marine_fish)]=['Sole fish (Cynoglossusarel) ', 80.1, 19.0, 1.0, 1.1, 0, 0, 367.0]
marine_fish.loc[len(marine_fish)]=['Stingray (Dasyatis pastinaca) ', 75.6, 23.9, 1.1, 0.6, 0, 0, 408.0]
marine_fish.loc[len(marine_fish)]=['Tarlava (Drepane punctata) ', 76.7, 21.7, 1.1, 1.2, 0, 0, 414.0]
marine_fish.loc[len(marine_fish)]=['Tholam (Plectorhinchus schotaf) ', 76.2, 20.7, 1.0, 2.1, 0, 0, 432.0]
marine_fish.loc[len(marine_fish)]=['Tilapia (Oreochromisniloticus) ', 79.5, 18.4, 1.1, 1.0, 0, 0, 349.0]
marine_fish.loc[len(marine_fish)]=['Tuna (Euthynnus affinis) ', 72.1, 24.5, 1.1, 1.4, 0, 0, 470.0]
marine_fish.loc[len(marine_fish)]=['Tuna,striped (Katsuwonuspelamis) ', 77.7, 21.2, 0.8, 1.1, 0, 0, 403.0]
marine_fish.loc[len(marine_fish)]=['Valava (Chirocentrus nudus) ',  77.2, 21.8, 1.2, 0, 0, 1.0, 410.0]
marine_fish.loc[len(marine_fish)]=['Vanjaram (Scomberomorus commerson) ', 72.3, 22.2, 1.3, 5.1, 0, 0, 570.0]
marine_fish.loc[len(marine_fish)]=['Vela meen (Aprionvirescens) ', 71.8, 22.1, 1.3, 4.3, 0, 0, 537.0]
marine_fish.loc[len(marine_fish)]=['Vora (Siganus javus) ', 76.9, 20.1, 1.2, 2.1, 0, 0, 422.0]
marine_fish.loc[len(marine_fish)]=['Whaleshark (Galeocerdo cuvier) ', 77.5, 21.8, 1.4, 0.8, 0, 0, 401.0]
marine_fish.loc[len(marine_fish)]=['Xiphinis (Xiphiasgladius) ', 79.2, 19.8, 1.1, 0.8, 0, 0, 370.0]
marine_fish.loc[len(marine_fish)]=['Z.Eggs, Cat fish (Ompok bimaculatus) ', 69.3, 24.6, 1.5, 5.2, 0, 0, 600.0]

print(marine_fish.head())
marine_shellfish=pd.DataFrame(columns=dietary_cols)
marine_shellfish.loc[len(marine_shellfish)]=['Crab (Menippemercenaria) ', 79.7, 10.2, 1.7, 1.4, 0, 0, 343.0]
marine_shellfish.loc[len(marine_shellfish)]=['Crab, sea (Portunus sanguinolentus) ', 79.6, 15.3, 0.9, 0.6, 0, 0, 283.0]
marine_shellfish.loc[len(marine_shellfish)]=['Lobster, brown (Thenus orientalis) ', 81.4, 15.9, 1.2, 0.5, 0, 0, 292.0]
marine_shellfish.loc[len(marine_shellfish)]=['Lobster,kingsize (Thenus orientalis) ', 77.7, 18.5, 1.0, 0.7, 0, 0, 375.0]
marine_shellfish.loc[len(marine_shellfish)]=['Mud crab (Scylla tranquebarica) ', 83.1, 10.0, 2.6, 0.5, 0, 0, 190.0]
marine_shellfish.loc[len(marine_shellfish)]=['Oyster (Crassostrea sp.) ', 82.5, 9.5, 2.4, 2.4, 0, 0, 252.0]
marine_shellfish.loc[len(marine_shellfish)]=['Tiger prawns, brown (Solenocera crassicornis) ', 82.4, 14.8, 0.9, 0.5, 0, 0, 273.0]
marine_shellfish.loc[len(marine_shellfish)]=['Tiger Prawns, orange (Penaeus monodon) ', 81.4, 14.2, 0.8, 0.7, 0, 0, 270.0]

print(marine_shellfish.head())
marine_mollusks=pd.DataFrame(columns=dietary_cols)
marine_mollusks.loc[len(marine_mollusks)]=['Clam, greenshell (Perna viridis ) ', 80.0, 12.1, 2.4, 0.9, 0, 0, 243.0]
marine_mollusks.loc[len(marine_mollusks)]=['Clam, white shell, ribbed (Meretrixmeretrix) ', 80.2, 11.8, 0.8, 1.3, 0, 0, 250.0]
marine_mollusks.loc[len(marine_mollusks)]=['Octopus(Octopus vulgaris) ', 80.4, 14.7, 1.2, 1.1, 0, 0, 334.0]
marine_mollusks.loc[len(marine_mollusks)]=['Squid, black (Loligosp.) ', 80.5, 16.1, 0.9, 1.0, 0, 0, 335.0]
marine_mollusks.loc[len(marine_mollusks)]=['Squid, hard shell (Sepia pharaonis) ', 80.5, 16.8, 0.7, 0.9, 0, 0, 320.0]
marine_mollusks.loc[len(marine_mollusks)]=['Squid, red (Loligoduvaucelii) ', 80.8, 16.2, 1.1, 1.4, 0, 0, 329.0]
marine_mollusks.loc[len(marine_mollusks)]=['Squid, white, small (Uroteuthis duvauceli) ', 79.3, 17.4, 1.1, 1.1, 0, 0, 353.0]

print(marine_mollusks.head())
freshwater_fish=pd.DataFrame(columns=dietary_cols)
freshwater_fish.loc[len(freshwater_fish)]=['Cat fish (Tandanus tandanus) ', 77.2, 15.8, 0.9, 6.2, 0, 0, 518.0]
freshwater_fish.loc[len(freshwater_fish)]=['Catla (Catla catla) ', 78.4, 17.9, 0.9, 2.1, 0, 0, 394.0]
freshwater_fish.loc[len(freshwater_fish)]=['Freshwater Eel (Anguilla anguilla) ', 75.5, 20.4, 1.0, 2.6, 0, 0, 451.0]
freshwater_fish.loc[len(freshwater_fish)]=['Goldfish (Carassiusauratus) ', 79.0, 16.9, 1.1, 2.9, 0, 0, 396.0]
freshwater_fish.loc[len(freshwater_fish)]=['Pangas (Pangasianodon hypophthalmus) ', 68.5, 17.1, 1.1, 16.7, 0, 0, 852.0]
freshwater_fish.loc[len(freshwater_fish)]=['Rohu (Labeorohita) ', 76.3, 19.7, 1.2, 2.3, 0, 0, 428.0]
freshwater_fish.loc[len(freshwater_fish)]=['Crab (Pachygrapsussp.) ', 80.5, 13.2, 1.2, 0.8, 0, 0, 327.0]
freshwater_fish.loc[len(freshwater_fish)]=['Prawns, big (Macrobrachium rosenbergii) ', 77.4, 19.2, 0.8, 0.5, 0, 0, 380.0]
freshwater_fish.loc[len(freshwater_fish)]=['Prawns,small (Macrobrachiumsp.) ', 82.5, 13.0, 0.8, 0.7, 0, 0, 297.0]
freshwater_fish.loc[len(freshwater_fish)]=['Tiger prawns (Macrobrachiumsp.) ', 83.2, 14.2, 0.8, 0.6, 0, 0, 284.0]

print(freshwater_fish.head())
dietary_element=[cereals_and_millets,freshwater_fish,marine_mollusks,marine_shellfish,marine_fish,animal,poultry,egg,milk,misc,mushroom,sugars,nuts,dry_spices,fresh_spices,roots,fruits,other_veg,green_leaf_veg,grains]
result=pd.concat(dietary_element)
print(result.head())
#print('Enter the element you want to search:')
search_element='Catla'#input()
search_element=search_element.lower()
match=[]
for index, row in result.iterrows():
    if search_element in row['Name'].lower():
        match.append(row['Name'])
if len(match)==0:
    print('Sorry, No match!')
elif len(match)==1:
    nutri_dict={}
    for each in dietary_cols:
        nutri_dict[each]=0
    for index, row in result.iterrows():
        if match[0] == row['Name']:
            for each in nutri_dict.keys():
                nutri_dict[each]=row[each]
    print(nutri_dict)
else:
    #print("Enter the number of your Element")
    for i in range (1,len(match)+1):
        print(str(i)+': '+match[i-1])
    ind=0#int(input())
    nutri_dict={}
    for each in dietary_cols:
        nutri_dict[each]=0
    for index, row in result.iterrows():
        if match[ind-1] == row['Name']:
            for each in nutri_dict.keys():
                nutri_dict[each]=row[each]
    print(nutri_dict)
def user_info():
    #print('Enter unit for weight')
    #print('1: kg, 2: lbs')
    wt=1#int(input())
    #print('Enter weight')
    if wt==1:
        weight=45#int(input())
    elif wt==2:
        weight=45#int(input())*0.453592
    else:
        #print('Sorry! You entered wrong unit')
        return
    #print('Enter unit for height')
    #print('1: cm, 2: ft inch')
    ht=2#int(input())
    #print('Enter height')
    if ht==1:
        height=122#int(input())/100
    elif ht==2:
        arr='5 1'.split()#input().split()
        height=int(arr[0])* 30.48+int(arr[1])*0.0833333* 30.48
        height=height/100
    else:
        #print('Sorry! You entered wrong unit')
        return
    #print('Enter age')
    age=22#int(input())
    #print('Enter sex')
    #print('1: Female, 2: Male')
    sex=1#int(input())
    if sex!=1 and sex!=2:
        #print('Sorry! You entered wrong unit')
        return
    #print('Enter PA according to below information')
    #print('1) sedentary – includes only the light physical activity associated with typical day-to-day life; (2) low active – adds 30 minutes per day walk at a speed of 4 miles per hour (mph); (3) active – adds an hourly moderate daily exercise; and (4) very active – includes vigorous daily exercise.')
    pa=1#int(input())
    if pa<1 or pa>4:
        #print('Sorry! wrong info')
        return
    #print('Are you under stress, pregnant, recovering from an illness, or if you are involved in consistent and intense weight or endurance training?')
    #print('1:Yes, 2: No')
    upper=2#int(input())
    if upper==1:
        protein=weight*1.7
    else:
        protein=weight*0.8
    m={}
    m['Protein']=protein
    if sex==1:
        if pa==1:
            PA=1.0
        elif pa==2:
            PA=1.14
        elif pa==3:
            PA=1.27
        else:
            PA=1.45
        m['TEE']=655 + (4.3* weight*2.20462) + (4.7 * height *39.3701) - (4.7 * age)*PA
    else:
        if pa==1:
            PA=1
        elif pa==2:
            PA=1.12
        elif pa==3:
            PA=1.27
        else:
            PA=1.54
        m['TEE']= 66 + (6.3* weight*2.20462) + (12.9 * height *39.3701) - (6.8 * age)*PA
    m['Fat']=m['TEE']*0.27/9
    m['Fibre']=m['TEE']*0.014
    m['Carbohydrate']=m['TEE']*0.55/4   
    return m
TEE=user_info()
print('Your needs')
print(TEE)
#meal array
meal=['Bajra (Pennisetum typhoideum)', 'Wheat flour, refined (Triticum aestivum) ', 'Soya bean, white (Glycine max) ', 'Apple, big (Malus domestica) ', 'Banana, ripe, montham (Musa x paradisiaca) ', 'Jowar (Sorghum vulgare) ', 'Mango, ripe, banganapalli (Mangifera indica) ', 'Z.Eggs, Cat fish (Ompok bimaculatus) ']
quantity=[14, 34, 54, 14, 76, 43, 12, 43]
'''
while (True):
    print('Enter Y if you want to add a food item')
    inp=input()
    if inp.lower()!='y':
        break
    print('Enter the element you want to add:')
    search_element=input()
    search_element=search_element.lower()
    match=[]
    for index, row in result.iterrows():
        if search_element in row['Name'].lower():
            match.append(row['Name'])
    if len(match)==0:
        print('Sorry, No match!')
    elif len(match)==1:
        meal.append(match[0])
        print('Enter Quantity in g')
        quan=int(input())
        quantity.append(quan)
    else:
        print("Enter the number of your Element")
        for i in range (1,len(match)+1):
            print(str(i)+': '+match[i-1])
        ind=int(input())
        meal.append(match[ind-1])
        print('Enter Quantity in g')
        quan=int(input())
        quantity.append(quan)
'''
requirement=TEE
available={}
for each in requirement.keys():
    available[each]=0
for i in range(0,len(meal)):
    for index, row in result.iterrows():
        if meal[i] == row['Name']:
            for each2 in nutri_dict.keys():
                if each2.lower()=='protein':
                    temp=row[each2]
                    available['Protein']+=quantity[i]*temp/100
                elif each2.lower()=='dietary fibre':
                    temp=row[each2]
                    available['Fibre']+=quantity[i]*temp/100
                elif 'fat' in each2.lower():
                    temp=row[each2]
                    available['Fat']+=quantity[i]*temp/100
                elif 'carb' in each2.lower():
                    temp=row[each2]
                    available['Carbohydrate']+=quantity[i]*temp/100
                elif 'energy' in each2.lower():
                    temp=row[each2]
                    available['TEE']+=quantity[i]*temp/(100*4.184)
print(available)
def analysis_of_completion(req,avail):
    
    for each in req.keys():
        f = FloatProgress(min=0, max=100,description=each)
        f.value=avail[each]*100/req[each]
        if f.value==100:
            f.bar_style='success'
        elif f.value<50 or f.value>150:
            f.bar_style='danger'
        elif f.value<75:
            f.bar_style='warning'
        else:
            f.bar_style='info'
        display(f)
analysis_of_completion(requirement,available)
        