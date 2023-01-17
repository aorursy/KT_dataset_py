library(readr)

library(dplyr)

library(ggplot2)

library(scales)

library(treemap)



system("ls ../input")
train <- read_csv("../input/train.csv")

#train <- train %>% sample_frac(0.001)

client <- read_csv("../input/cliente_tabla.csv")

product <- read_csv("../input/producto_tabla.csv")

town <- read_csv("../input/town_state.csv")
