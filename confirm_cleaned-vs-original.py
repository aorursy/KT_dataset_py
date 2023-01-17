import tempfile

import pandas as pd

pd.set_option("display.max_colwidth", 1000)





# Original (https://github.com/cvdfoundation/google-landmark)

train_partial_csv = """

id,url,landmark_id

7e4627eb6f721857,https://upload.wikimedia.org/wikipedia/commons/6/6d/Lateral_view_of_one_of_the_buildings.JPG,140690

5366e61022b7dc55,https://upload.wikimedia.org/wikipedia/commons/6/6a/Pantallas_interactivas_en_el_MIDE.jpg,140690

39e4baa4e58810ac,https://upload.wikimedia.org/wikipedia/commons/7/7f/The_Reaper_%2814069069341%29.jpg,165900

140690de98deb562,https://upload.wikimedia.org/wikipedia/commons/f/f1/Ormont_%28Eifel%29%3B_katholische_Pfarrkirche_St._Margaretha_h.jpg,137389

67a7f496d48fbc70,https://upload.wikimedia.org/wikipedia/commons/a/a8/Banco_de_M%C3%A9xico%27s_section.JPG,140690

530535d32be0afea,http://upload.wikimedia.org/wikipedia/commons/0/01/Simulador_de_mercado_en_el_MIDE.jpg,140690

9706dec8f3416a90,https://upload.wikimedia.org/wikipedia/commons/b/be/Coins_and_display.JPG,140690

93baea87be87591d,https://upload.wikimedia.org/wikipedia/commons/6/6c/Garden_closeup.JPG,140690

9f4d438d57feefad,http://upload.wikimedia.org/wikipedia/commons/6/6a/Entrada_al_MIDE.jpg,140690

be20cbb708cd2ecc,https://upload.wikimedia.org/wikipedia/commons/c/c0/Display_of_values.JPG,140690

4540a114069023e1,https://upload.wikimedia.org/wikipedia/commons/9/9d/SMV_High_St.jpg,196741

cc46f5ba5934dbf1,https://upload.wikimedia.org/wikipedia/commons/a/a7/Medio_ambiente_y_ciudad.JPG,140690

bbca62964af017c9,https://upload.wikimedia.org/wikipedia/commons/b/b1/325_Shrine_Mont_Tour_%2810061406905%29.jpg,199506

6808120688d39562,https://upload.wikimedia.org/wikipedia/commons/a/a1/Historical_Coins.JPG,140690

3ffa0a133c69b51e,https://upload.wikimedia.org/wikipedia/commons/d/d8/Ex_Convento_de_Betlemitas.jpg,140690

b7cd1c698986ec7d,https://upload.wikimedia.org/wikipedia/commons/1/16/Income_Map.JPG,140690

6569e5b081b2e990,https://upload.wikimedia.org/wikipedia/commons/e/ea/Banco_de_M%C3%A9xico%27s_logo.JPG,140690

6d432b43959508c3,http://upload.wikimedia.org/wikipedia/commons/e/e0/Salas_del_Museo-Interactivo-de-Econom%C3%ADa.jpg,140690

b5ccdb7f54a788a3,https://upload.wikimedia.org/wikipedia/commons/1/10/Lateral_hall.JPG,140690

7f6f5f736985d7d3,https://upload.wikimedia.org/wikipedia/commons/7/7c/Garden_on_top.JPG,140690

f3214d10004d0fe4,https://upload.wikimedia.org/wikipedia/commons/1/1c/Garden_and_sculpture.JPG,140690

9c1bf5cdd65d774f,https://upload.wikimedia.org/wikipedia/commons/4/44/Garden_on_the_rooftop.JPG,140690

06b826af914a6e30,https://upload.wikimedia.org/wikipedia/commons/b/b8/Maquina_de_Imprenta_antigua.JPG,140690

795c875e2cff017a,https://upload.wikimedia.org/wikipedia/commons/3/39/Bike_and_pollution.JPG,140690

041fed0882eb3da8,https://upload.wikimedia.org/wikipedia/commons/1/14/Entrada_exconvento_sombras.JPG,140690

58a68053061b2bf6,https://upload.wikimedia.org/wikipedia/commons/d/dd/Mexican_bill.JPG,140690

a2ab900394464976,https://upload.wikimedia.org/wikipedia/commons/b/be/Ventana_y_azulejos.JPG,140690

d7be3c582cc7eb7d,https://upload.wikimedia.org/wikipedia/commons/6/66/Display_of_bills.JPG,140690

a6370306caf4af52,https://upload.wikimedia.org/wikipedia/commons/a/a6/Horizontal_garden.JPG,140690

4b6d79211cb5855f,https://upload.wikimedia.org/wikipedia/commons/5/52/Lot_of_bill_paper.JPG,140690

eb7da1441ab9f78f,http://upload.wikimedia.org/wikipedia/commons/9/91/MIDE.JPG,140690

73541aafdbb2ddf4,https://upload.wikimedia.org/wikipedia/commons/9/97/Main_entrance_Hall.JPG,140690

f95b9d6065ff8725,https://upload.wikimedia.org/wikipedia/commons/7/7c/Nose_of_Grumman_S-2A_Tracker_%E2%80%98624%E2%80%99_%28136624%29_%2829914069024%29.jpg,202062

c118f1004da6edf9,https://upload.wikimedia.org/wikipedia/commons/c/cc/Cactus_and_interior_of_rooftop_garden.JPG,140690

f80140690052ae41,https://upload.wikimedia.org/wikipedia/commons/4/4a/%D0%9E%D0%B4%D0%B5%D1%81%D0%B0%2C_%D0%91%D1%83%D0%B4%D1%96%D0%B2%D0%BB%D1%8F_%D0%86%D0%BB%D0%BB%D1%96%D0%BD%D1%81%D1%8C%D0%BA%D0%BE%D1%97_%D1%86%D0%B5%D1%80%D0%BA%D0%B2%D0%B8_%D0%90%D1%84%D0%BE%D0%BD%D1%81%D1%8C%D0%BA%D0%BE%D0%B3%D0%BE_%D0%BF%D0%BE%D0%B4%D0%B2%D1%96%D1%80%27%D1%8F%2C_%D0%B2%D1%83%D0%BB._%D0%9F%D1%83%D1%88%D0%BA%D1%96%D0%BD%D1%81%D1%8C%D0%BA%D0%B0_79.jpg,164047

2dcf7696c5fbddf5,https://upload.wikimedia.org/wikipedia/commons/7/7e/Porthcurno_beach_-_geograph.org.uk_-_1406907.jpg,118158

6aaa84dbc4d926d5,https://upload.wikimedia.org/wikipedia/commons/a/ab/Bethlemite_Convent_from_Tacuba.jpg,140690

fde6f3cc1478879f,https://upload.wikimedia.org/wikipedia/commons/e/e7/Credit_card_display.JPG,140690

30086d6a282b12fa,https://upload.wikimedia.org/wikipedia/commons/d/d3/Betlemitas_de_la_ciudad.jpg,140690

f39a04040d6f16ae,https://upload.wikimedia.org/wikipedia/commons/1/14/Display_of_objects.JPG,140690

4d8dd97bf586be2c,https://upload.wikimedia.org/wikipedia/commons/d/db/Common_objects.JPG,140690

1a3041db36db53c0,https://upload.wikimedia.org/wikipedia/commons/1/1b/Detalle_azulejos.JPG,140690

ea620417fbfdc62b,https://upload.wikimedia.org/wikipedia/commons/8/8d/Banco_de_Mexico%27s_history.JPG,140690

80050d8c6be97cbe,https://upload.wikimedia.org/wikipedia/commons/6/64/MIDE_Museum_Mexico.JPG,140690

"""

with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:

    f.write(train_partial_csv.strip())

    f.flush()

    df_orig = pd.read_csv(f.name)
df = pd.read_csv('../input/cleaned_subsets_train2019/train19_cleaned_verifythresh20_freqthresh3.csv')

df.head()
df_cleaned_MIDE = pd.DataFrame({'id': df[df.landmark_id == 140690].iloc[0].images.split(' ')})

df_cleaned_MIDE = df_cleaned_MIDE.merge(df_orig, how='left', on='id')

df_cleaned_MIDE.head()
df_original_MIDE = df_orig[df_orig.landmark_id == 140690].copy()

df_original_MIDE.head()